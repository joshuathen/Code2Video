from manim import *
import numpy as np

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

class Section4Scene(TeachingScene):
    def construct(self):
        title_text = "Step 2: The Digital Handshake (Whispering in the Crowd)"
        lecture_lines = [
            "- Alice and Bob's phones exchange random RPIs via Bluetooth.",
            "- These IDs look like noise to any outside observer.",
            "- Each phone keeps a private diary of IDs seen nearby.",
            "- No location data or names are ever exchanged or saved.",
            "- The data stays on the device and is never uploaded."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Phone A (#0000FF) sends a blue dot to Phone B (#00FF00), while Phone B sends a green dot to Phone A.
        self.lecture[0].set_color(BLUE)
        
        phone_a = RoundedRectangle(corner_radius=0.1, height=1.2, width=0.7, color=BLUE)
        phone_b = RoundedRectangle(corner_radius=0.1, height=1.2, width=0.7, color=GREEN)
        
        self.place_at_grid(phone_a, 'B2')
        self.place_at_grid(phone_b, 'B5')
        
        alice_label = Text("Alice", font_size=18, color=BLUE)
        bob_label = Text("Bob", font_size=18, color=GREEN)
        self.place_at_grid(alice_label, 'C2')
        self.place_at_grid(bob_label, 'C5')
        
        dot_a = Dot(color=BLUE)
        dot_b = Dot(color=GREEN)
        
        dot_a.move_to(phone_a.get_center())
        dot_b.move_to(phone_b.get_center())
        
        self.add(phone_a, phone_b, alice_label, bob_label)
        self.play(FadeIn(dot_a), FadeIn(dot_b))
        
        path_a_to_b = Line(phone_a.get_center(), phone_b.get_center())
        path_b_to_a = Line(phone_b.get_center(), phone_a.get_center())
        
        self.play(
            dot_a.animate.move_to(path_a_to_b.point_from_proportion(0.5)),
            dot_b.animate.move_to(path_b_to_a.point_from_proportion(0.5)),
            run_time=1.5
        )

        # === Animation for Lecture Line 2 ===
        # The dots transform into grey random characters like '7x!9?' (#AAAAAA) as they travel between phones.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(GRAY)
        
        char_a = Text("7x!9?", font_size=20, color="#AAAAAA")
        char_b = Text("k2#p0", font_size=20, color="#AAAAAA")
        
        char_a.move_to(dot_a.get_center())
        char_b.move_to(dot_b.get_center())

        self.play(
            ReplacementTransform(dot_a, char_a),
            ReplacementTransform(dot_b, char_b),
            run_time=0.5
        )
        
        self.play(
            char_a.animate.move_to(phone_b.get_center()),
            char_b.animate.move_to(phone_a.get_center()),
            run_time=1.5
        )

        # === Animation for Lecture Line 3 ===
        # Both phones show a white 'Diary' icon (#FFFFFF); Alice's diary stores the green dot.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(WHITE)
        
        diary_a = VGroup(
            Square(side_length=0.6, color=WHITE),
            Line([-0.2, 0.1, 0], [0.2, 0.1, 0], stroke_width=2),
            Line([-0.2, -0.1, 0], [0.2, -0.1, 0], stroke_width=2)
        )
        diary_b = diary_a.copy()
        
        self.place_at_grid(diary_a, 'D2')
        self.place_at_grid(diary_b, 'D5')
        
        diary_label_a = Text("Diary", font_size=16, color=WHITE)
        diary_label_b = Text("Diary", font_size=16, color=WHITE)
        self.place_at_grid(diary_label_a, 'E2')
        self.place_at_grid(diary_label_b, 'E5')

        self.play(
            FadeIn(diary_a), FadeIn(diary_b),
            FadeIn(diary_label_a), FadeIn(diary_label_b),
            char_a.animate.move_to(diary_b.get_center()).scale(0.5),
            char_b.animate.move_to(diary_a.get_center()).scale(0.5),
        )

        # === Animation for Lecture Line 4 ===
        # Red text 'GPS: OFF' and 'NAME: HIDDEN' (#FF0000) appears above both phones.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(RED)
        
        gps_a = Text("GPS: OFF", font_size=16, color=RED)
        name_a = Text("NAME: HIDDEN", font_size=16, color=RED)
        gps_b = Text("GPS: OFF", font_size=16, color=RED)
        name_b = Text("NAME: HIDDEN", font_size=16, color=RED)
        
        self.place_at_grid(gps_a, 'A2')
        name_a.next_to(gps_a, DOWN, buff=0.1)
        self.place_at_grid(gps_b, 'A5')
        name_b.next_to(gps_b, DOWN, buff=0.1)

        self.play(
            Write(gps_a), Write(name_a),
            Write(gps_b), Write(name_b)
        )

        # === Animation for Lecture Line 5 ===
        # An arrow points from the Diary to the phone's center; a distant white 'Cloud' (#FFFFFF) is crossed out in red.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(WHITE)
        
        arrow_a = Arrow(start=diary_a.get_top(), end=phone_a.get_bottom(), color=WHITE, buff=0.1, stroke_width=3)
        arrow_b = Arrow(start=diary_b.get_top(), end=phone_b.get_bottom(), color=WHITE, buff=0.1, stroke_width=3)
        
        cloud = VGroup(
            Circle(radius=0.3, color=WHITE),
            Circle(radius=0.2, color=WHITE).shift(LEFT*0.3),
            Circle(radius=0.2, color=WHITE).shift(RIGHT*0.3)
        ).set_fill(WHITE, opacity=0.3)
        
        self.place_in_area(cloud, 'C3', 'D4', scale_factor=0.8)
        cross = Cross(cloud, color=RED)
        
        self.play(
            Create(arrow_a), Create(arrow_b),
            FadeIn(cloud),
            Create(cross)
        )
        self.wait(2)
