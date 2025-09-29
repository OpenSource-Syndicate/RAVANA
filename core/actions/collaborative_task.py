import logging
from typing import Dict, Any, List
from core.actions.action import Action

logger = logging.getLogger(__name__)


class CollaborativeTaskAction(Action):
    """Action for managing collaborative tasks between RAVANA and users with feedback mechanisms."""

    def __init__(self, system=None, data_service=None):
        # Store system and data_service references
        self.system = system
        self.data_service = data_service
        self.collaborative_tasks = {}  # In-memory storage for collaborative tasks
        self.task_feedback = {}  # Storage for task feedback

    @property
    def name(self) -> str:
        return "collaborative_task"

    @property
    def description(self) -> str:
        return "Manage collaborative tasks between RAVANA and users, including creation, tracking, completion, and feedback collection"

    @property
    def parameters(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "task_type",
                "type": "string",
                "description": "Type of collaborative task (create, update, complete, cancel, provide_feedback, request_feedback)"
            },
            {
                "name": "task_id",
                "type": "string",
                "description": "Unique identifier for the task (required for update, complete, cancel, provide_feedback)"
            },
            {
                "name": "title",
                "type": "string",
                "description": "Title of the task (required for create)"
            },
            {
                "name": "description",
                "type": "string",
                "description": "Detailed description of the task"
            },
            {
                "name": "user_id",
                "type": "string",
                "description": "User ID to collaborate with"
            },
            {
                "name": "priority",
                "type": "string",
                "description": "Priority level (low, medium, high, critical)"
            },
            {
                "name": "deadline",
                "type": "string",
                "description": "Deadline for task completion (ISO format)"
            },
            {
                "name": "feedback",
                "type": "string",
                "description": "Feedback content (required for provide_feedback)"
            },
            {
                "name": "feedback_type",
                "type": "string",
                "description": "Type of feedback (positive, negative, suggestion, question)"
            },
            {
                "name": "rating",
                "type": "integer",
                "description": "Numerical rating for the task (1-10)"
            }
        ]

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the collaborative task action.

        Args:
            **kwargs: Action parameters

        Returns:
            Dictionary with execution result
        """
        try:
            task_type = kwargs.get("task_type", "create")
            task_id = kwargs.get("task_id")
            title = kwargs.get("title")
            description = kwargs.get("description")
            user_id = kwargs.get("user_id")
            priority = kwargs.get("priority", "medium")
            deadline = kwargs.get("deadline")
            feedback = kwargs.get("feedback")
            feedback_type = kwargs.get("feedback_type", "suggestion")
            rating = kwargs.get("rating", 5)

            if task_type == "create":
                return await self._create_task(title, description, user_id, priority, deadline)
            elif task_type == "update":
                return await self._update_task(task_id, title, description, priority, deadline)
            elif task_type == "complete":
                return await self._complete_task(task_id)
            elif task_type == "cancel":
                return await self._cancel_task(task_id)
            elif task_type == "provide_feedback":
                return await self._provide_feedback(task_id, user_id, feedback, feedback_type, rating)
            elif task_type == "request_feedback":
                return await self._request_feedback(task_id, user_id)
            else:
                return {"error": f"Unknown task type: {task_type}"}

        except Exception as e:
            logger.error(f"Error executing collaborative task action: {e}")
            return {"error": str(e)}

    async def _create_task(self, title: str, description: str, user_id: str,
                           priority: str, deadline: str) -> Dict[str, Any]:
        """
        Create a new collaborative task.

        Args:
            title: Task title
            description: Task description
            user_id: User ID to collaborate with
            priority: Task priority
            deadline: Task deadline

        Returns:
            Dictionary with creation result
        """
        try:
            import uuid
            from datetime import datetime

            # Generate a unique task ID
            task_id = str(uuid.uuid4())

            # Create task object
            task = {
                "task_id": task_id,
                "title": title,
                "description": description,
                "user_id": user_id,
                "priority": priority,
                "deadline": deadline,
                "status": "pending",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "progress_updates": [],
                "collaboration_history": []
            }

            # Store task
            self.collaborative_tasks[task_id] = task

            # Add to collaboration history
            task["collaboration_history"].append({
                "event": "task_created",
                "timestamp": datetime.now().isoformat(),
                "details": f"Task '{title}' created"
            })

            # Notify user about the new task (if system is available)
            if self.system and hasattr(self.system, 'conversational_ai'):
                try:
                    message = f"I've created a collaborative task for us: '{title}'. Would you like to work on this together? You can provide feedback on this task at any time using the 'provide_feedback' action."
                    await self.system.conversational_ai.send_message_to_user(user_id, message)
                except Exception as e:
                    logger.error(f"Failed to notify user about new task: {e}")

            logger.info(f"Created collaborative task {task_id}: {title}")
            return {
                "result": "success",
                "task_id": task_id,
                "message": f"Collaborative task '{title}' created successfully"
            }

        except Exception as e:
            logger.error(f"Error creating collaborative task: {e}")
            return {"error": f"Failed to create collaborative task: {e}"}

    async def _update_task(self, task_id: str, title: str = None, description: str = None,
                           priority: str = None, deadline: str = None) -> Dict[str, Any]:
        """
        Update an existing collaborative task.

        Args:
            task_id: Task ID to update
            title: New task title (optional)
            description: New task description (optional)
            priority: New task priority (optional)
            deadline: New task deadline (optional)

        Returns:
            Dictionary with update result
        """
        try:
            from datetime import datetime

            # Check if task exists
            if task_id not in self.collaborative_tasks:
                return {"error": f"Task {task_id} not found"}

            # Update task fields
            task = self.collaborative_tasks[task_id]
            if title:
                task["title"] = title
            if description:
                task["description"] = description
            if priority:
                task["priority"] = priority
            if deadline:
                task["deadline"] = deadline

            task["updated_at"] = datetime.now().isoformat()

            # Add to collaboration history
            task["collaboration_history"].append({
                "event": "task_updated",
                "timestamp": datetime.now().isoformat(),
                "details": f"Task updated with new information"
            })

            logger.info(f"Updated collaborative task {task_id}")
            return {
                "result": "success",
                "task_id": task_id,
                "message": f"Collaborative task '{task['title']}' updated successfully"
            }

        except Exception as e:
            logger.error(f"Error updating collaborative task {task_id}: {e}")
            return {"error": f"Failed to update collaborative task: {e}"}

    async def _complete_task(self, task_id: str) -> Dict[str, Any]:
        """
        Mark a collaborative task as completed.

        Args:
            task_id: Task ID to complete

        Returns:
            Dictionary with completion result
        """
        try:
            from datetime import datetime

            # Check if task exists
            if task_id not in self.collaborative_tasks:
                return {"error": f"Task {task_id} not found"}

            # Update task status
            task = self.collaborative_tasks[task_id]
            task["status"] = "completed"
            task["updated_at"] = datetime.now().isoformat()

            # Add to collaboration history
            task["collaboration_history"].append({
                "event": "task_completed",
                "timestamp": datetime.now().isoformat(),
                "details": f"Task '{task['title']}' marked as completed"
            })

            # Notify user about task completion (if system is available)
            if self.system and hasattr(self.system, 'conversational_ai'):
                try:
                    message = f"Great job! We've completed the collaborative task: '{task['title']}'. I'd appreciate your feedback on how this collaboration went. You can use the 'provide_feedback' action to share your thoughts."
                    await self.system.conversational_ai.send_message_to_user(task["user_id"], message)
                except Exception as e:
                    logger.error(
                        f"Failed to notify user about task completion: {e}")

            logger.info(f"Completed collaborative task {task_id}")
            return {
                "result": "success",
                "task_id": task_id,
                "message": f"Collaborative task '{task['title']}' completed successfully"
            }

        except Exception as e:
            logger.error(f"Error completing collaborative task {task_id}: {e}")
            return {"error": f"Failed to complete collaborative task: {e}"}

    async def _cancel_task(self, task_id: str) -> Dict[str, Any]:
        """
        Cancel a collaborative task.

        Args:
            task_id: Task ID to cancel

        Returns:
            Dictionary with cancellation result
        """
        try:
            from datetime import datetime

            # Check if task exists
            if task_id not in self.collaborative_tasks:
                return {"error": f"Task {task_id} not found"}

            # Update task status
            task = self.collaborative_tasks[task_id]
            task["status"] = "cancelled"
            task["updated_at"] = datetime.now().isoformat()

            # Add to collaboration history
            task["collaboration_history"].append({
                "event": "task_cancelled",
                "timestamp": datetime.now().isoformat(),
                "details": f"Task '{task['title']}' cancelled"
            })

            # Notify user about task cancellation (if system is available)
            if self.system and hasattr(self.system, 'conversational_ai'):
                try:
                    message = f"I'm cancelling our collaborative task: '{task['title']}'. Let me know if you'd like to work on something else. Your feedback on why this task was cancelled would be valuable."
                    await self.system.conversational_ai.send_message_to_user(task["user_id"], message)
                except Exception as e:
                    logger.error(
                        f"Failed to notify user about task cancellation: {e}")

            logger.info(f"Cancelled collaborative task {task_id}")
            return {
                "result": "success",
                "task_id": task_id,
                "message": f"Collaborative task '{task['title']}' cancelled successfully"
            }

        except Exception as e:
            logger.error(f"Error cancelling collaborative task {task_id}: {e}")
            return {"error": f"Failed to cancel collaborative task: {e}"}

    async def _provide_feedback(self, task_id: str, user_id: str, feedback: str,
                                feedback_type: str, rating: int) -> Dict[str, Any]:
        """
        Provide feedback on a collaborative task.

        Args:
            task_id: Task ID to provide feedback on
            user_id: User ID providing feedback
            feedback: Feedback content
            feedback_type: Type of feedback
            rating: Numerical rating (1-10)

        Returns:
            Dictionary with feedback result
        """
        try:
            from datetime import datetime

            # Validate rating
            if not 1 <= rating <= 10:
                return {"error": "Rating must be between 1 and 10"}

            # Check if task exists
            if task_id not in self.collaborative_tasks:
                return {"error": f"Task {task_id} not found"}

            # Create feedback entry
            feedback_entry = {
                "feedback_id": str(uuid.uuid4()),
                "task_id": task_id,
                "user_id": user_id,
                "feedback": feedback,
                "feedback_type": feedback_type,
                "rating": rating,
                "timestamp": datetime.now().isoformat()
            }

            # Store feedback
            if task_id not in self.task_feedback:
                self.task_feedback[task_id] = []
            self.task_feedback[task_id].append(feedback_entry)

            # Add to collaboration history
            task = self.collaborative_tasks[task_id]
            task["collaboration_history"].append({
                "event": "feedback_provided",
                "timestamp": datetime.now().isoformat(),
                "details": f"User provided {feedback_type} feedback with rating {rating}"
            })

            # Notify system about feedback (if system is available)
            if self.system and hasattr(self.system, 'conversational_ai'):
                try:
                    message = f"Thank you for your feedback on task '{task['title']}'. I'll use this information to improve our future collaborations."
                    await self.system.conversational_ai.send_message_to_user(user_id, message)
                except Exception as e:
                    logger.error(
                        f"Failed to notify user about feedback receipt: {e}")

            logger.info(f"Received feedback for collaborative task {task_id}")
            return {
                "result": "success",
                "feedback_id": feedback_entry["feedback_id"],
                "message": "Feedback recorded successfully"
            }

        except Exception as e:
            logger.error(f"Error providing feedback for task {task_id}: {e}")
            return {"error": f"Failed to record feedback: {e}"}

    async def _request_feedback(self, task_id: str, user_id: str) -> Dict[str, Any]:
        """
        Request feedback on a collaborative task.

        Args:
            task_id: Task ID to request feedback on
            user_id: User ID to request feedback from

        Returns:
            Dictionary with request result
        """
        try:
            # Check if task exists
            if task_id not in self.collaborative_tasks:
                return {"error": f"Task {task_id} not found"}

            # Get task
            task = self.collaborative_tasks[task_id]

            # Notify user to provide feedback (if system is available)
            if self.system and hasattr(self.system, 'conversational_ai'):
                try:
                    message = f"I'd like to get your feedback on our collaborative task: '{task['title']}'. Could you please share your thoughts on how this collaboration went? You can use the 'provide_feedback' action to give me your feedback."
                    await self.system.conversational_ai.send_message_to_user(user_id, message)
                except Exception as e:
                    logger.error(f"Failed to request feedback from user: {e}")

            logger.info(f"Requested feedback for collaborative task {task_id}")
            return {
                "result": "success",
                "message": "Feedback request sent to user"
            }

        except Exception as e:
            logger.error(f"Error requesting feedback for task {task_id}: {e}")
            return {"error": f"Failed to request feedback: {e}"}

    def get_task(self, task_id: str) -> Dict[str, Any]:
        """
        Get a collaborative task by ID.

        Args:
            task_id: Task ID to retrieve

        Returns:
            Task dictionary or None if not found
        """
        return self.collaborative_tasks.get(task_id)

    def get_user_tasks(self, user_id: str) -> list:
        """
        Get all collaborative tasks for a user.

        Args:
            user_id: User ID to retrieve tasks for

        Returns:
            List of tasks for the user
        """
        return [task for task in self.collaborative_tasks.values() if task.get("user_id") == user_id]

    def get_task_feedback(self, task_id: str) -> List[Dict[str, Any]]:
        """
        Get all feedback for a task.

        Args:
            task_id: Task ID to retrieve feedback for

        Returns:
            List of feedback entries for the task
        """
        return self.task_feedback.get(task_id, [])

    def get_user_feedback(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all feedback provided by a user.

        Args:
            user_id: User ID to retrieve feedback for

        Returns:
            List of feedback entries from the user
        """
        user_feedback = []
        for task_feedback_list in self.task_feedback.values():
            for feedback in task_feedback_list:
                if feedback.get("user_id") == user_id:
                    user_feedback.append(feedback)
        return user_feedback

    async def analyze_collaboration_patterns(self) -> Dict[str, Any]:
        """
        Analyze collaboration patterns to identify improvement opportunities.

        Returns:
            Dictionary with analysis results
        """
        try:
            # Collect statistics
            total_tasks = len(self.collaborative_tasks)
            completed_tasks = len(
                [t for t in self.collaborative_tasks.values() if t.get("status") == "completed"])
            cancelled_tasks = len(
                [t for t in self.collaborative_tasks.values() if t.get("status") == "cancelled"])
            total_feedback = sum(len(feedback_list)
                                 for feedback_list in self.task_feedback.values())

            # Calculate completion rate
            completion_rate = completed_tasks / total_tasks if total_tasks > 0 else 0

            # Calculate average rating from feedback
            all_ratings = [f.get("rating", 0) for feedback_list in self.task_feedback.values(
            ) for f in feedback_list]
            average_rating = sum(all_ratings) / \
                len(all_ratings) if all_ratings else 0

            # Identify feedback types
            feedback_types = {}
            for feedback_list in self.task_feedback.values():
                for feedback in feedback_list:
                    feedback_type = feedback.get("feedback_type", "unknown")
                    feedback_types[feedback_type] = feedback_types.get(
                        feedback_type, 0) + 1

            analysis = {
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "cancelled_tasks": cancelled_tasks,
                "completion_rate": completion_rate,
                "total_feedback": total_feedback,
                "average_rating": average_rating,
                "feedback_types": feedback_types,
                "most_active_users": self._get_most_active_users()
            }

            logger.info(
                f"Analyzed collaboration patterns: {completed_tasks}/{total_tasks} tasks completed")
            return analysis

        except Exception as e:
            logger.error(f"Error analyzing collaboration patterns: {e}")
            return {"error": f"Failed to analyze collaboration patterns: {e}"}

    def _get_most_active_users(self) -> Dict[str, int]:
        """
        Get the most active users in collaborations.

        Returns:
            Dictionary mapping user IDs to task counts
        """
        user_task_counts = {}
        for task in self.collaborative_tasks.values():
            user_id = task.get("user_id")
            if user_id:
                user_task_counts[user_id] = user_task_counts.get(
                    user_id, 0) + 1
        return user_task_counts
