from manim import *

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section5Scene(TeachingScene):
    def construct(self):
        # Setup
        lecture_lines = [
            "- After a positive test, Alice uploads her Secret Keys.",
            "- She only shares keys for her infectious period.",
            "- The server never learns who Alice actually met."
        ]
        self.setup_layout("Step 3: The Diagnosis and Decentralized Upload", lecture_lines)
        
        # Elements
        # Alice's Phone
        phone_body = RoundedRectangle(height=2.5, width=1.5, corner_radius=0.2, color="#5555FF", stroke_width=4)
        phone_screen = Rectangle(height=2.0, width=1.3, color="#000000", fill_opacity=1)
        phone = VGroup(phone_body, phone_screen)
        self.place_at_grid(phone, 'C2', scale_factor=0.9) # Resolved issue 36
        
        # Cloud (Public Server)
        cloud_circles = VGroup(
            Circle(radius=0.4, color="#FFFFFF", fill_opacity=1).shift(LEFT*0.3),
            Circle(radius=0.5, color="#FFFFFF", fill_opacity=1),
            Circle(radius=0.4, color="#FFFFFF", fill_opacity=1).shift(RIGHT*0.3),
            Circle(radius=0.35, color="#FFFFFF", fill_opacity=1).shift(UP*0.25)
        )
        cloud = cloud_circles
        self.place_at_grid(cloud, 'C6', scale_factor=1.0) # Resolved issue 35
        cloud_label = Text("Public Server", font_size=18, color="#FFFFFF").next_to(cloud, DOWN, buff=0.2)
        
        # Secret Key (SK_Alice)
        key_head = Circle(radius=0.1, color="#FFD700", fill_opacity=1)
        key_body = Rectangle(height=0.05, width=0.2, color="#FFD700", fill_opacity=1).next_to(key_head, RIGHT, buff=0)
        key_icon = VGroup(key_head, key_body)
        key_label = Text("SK_Alice", font_size=14, color="#FFD700").next_to(key_icon, UP, buff=0.1)
        key_group = VGroup(key_icon, key_label)
        
        # Encounter Log
        log_bg = Rectangle(height=1.0, width=1.4, color="#808080", fill_opacity=0.3)
        log_title = Text("Encounter Log", font_size=12, color="#808080").shift(UP*0.35)
        log_line_list = VGroup(*[Line(LEFT*0.5, RIGHT*0.5, stroke_width=1, color="#808080") for _ in range(3)]).arrange(DOWN, buff=0.1).shift(DOWN*0.1)
        log_group = VGroup(log_bg, log_title, log_line_list)
        self.place_at_grid(log_group, 'F2') # Resolved issue 34
        
        # Share Button
        share_btn_bg = RoundedRectangle(height=0.4, width=1.0, corner_radius=0.1, color="#00FF00", fill_opacity=1)
        share_btn_text = Text("SHARE", font_size=14, color="#000000")
        share_btn = VGroup(share_btn_bg, share_btn_text)
        self.place_at_grid(share_btn, 'D2', scale_factor=0.8)

        # Pre-animation state
        self.add(phone, log_group, cloud, cloud_label)
        
        # === Animation for Lecture Line 1 ===
        # Alice's phone #5555FF turns red #FF0000 and 'Share' button #00FF00 is pressed. Highlight lecture line 1.
        self.play(self.lecture[0].animate.set_color("#FF0000")) # Matching phone change
        self.play(
            phone_body.animate.set_color("#FF0000"),
            FadeIn(share_btn),
            run_time=1
        )
        self.play(
            share_btn.animate.scale(0.9),
            run_time=0.1,
            rate_func=there_and_back
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Move 'SK_Alice' key icon #FFD700 from phone to Cloud icon #FFFFFF. Highlight lecture line 2.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FFD700") # Matching key color
        )
        self.place_at_grid(key_group, 'C2', scale_factor=1.0)
        self.play(FadeIn(key_group))
        self.play(
            key_group.animate.move_to(self.grid['C6']), # Adjusted to cloud pos
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show 'Encounter Log' list #808080 staying on phone with crossed-out arrow to Cloud. Highlight lecture line 3.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#808080") # Matching log color
        )
        
        # Arrow from log to cloud
        # Use cloud.get_left() and log_group.get_right()
        arrow = Arrow(start=log_group.get_right(), end=cloud.get_bottom(), color="#FF0000", buff=0.1)
        cross_line1 = Line(UP + LEFT, DOWN + RIGHT, color="#FF0000").scale(0.2).move_to(arrow.get_center())
        cross_line2 = Line(UP + RIGHT, DOWN + LEFT, color="#FF0000").scale(0.2).move_to(arrow.get_center())
        cross = VGroup(cross_line1, cross_line2)
        
        self.play(Create(arrow))
        self.play(Create(cross))
        
        self.wait(2)
        
        # Final highlight removal
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(2)
