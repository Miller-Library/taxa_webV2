css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}

'''


bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://img.freepik.com/free-vector/cute-robot-holding-phone-with-laptop-cartoon-vector-icon-illustration-science-technology-isolated_138676-4870.jpg?w=740&t=st=1691702535~exp=1691703135~hmac=8af8d6813a33275c5acd28838caa6bb79b595bea37d94aa57a819287972ceb12" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://media.licdn.com/dms/image/C5603AQFyUeiUxTky0A/profile-displayphoto-shrink_200_200/0/1580955171316?e=1697068800&v=beta&t=R-6RUUCEfBkl0_HqAo9EHeeWkOKy8zpjZKU--H3f8oY">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''
