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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Proximity tracing faces privacy challenges.", "We must track exposure without identities.", "Central servers should not track individuals."]
        self.setup_layout("The Problem: Proximity vs. Privacy", lecture_lines)
        
        # Assets
        user_a = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/person.svg", color=WHITE)
        user_b = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/person.svg", color=WHITE)
        phone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/smartphone.svg", color=WHITE)
        label_a = Text("User A", font_size=18, color=WHITE)
        label_b = Text("User B", font_size=18, color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.place_at_grid(user_a, 'B2', scale_factor=0.8)
        self.place_at_grid(user_b, 'B5', scale_factor=0.8)
        self.add(user_a, user_b)
        label_a.next_to(user_a, DOWN)
        label_b.next_to(user_b, DOWN)
        self.add(label_a, label_b)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFFFF"))
        barrier = Line(self.grid['A3'], self.grid['F4'], color="#FF0000", stroke_width=4)
        
        self.play(Create(barrier))
        
        # Proximity scenario with phone
        self.place_at_grid(phone, 'C3', scale_factor=0.6)
        risk_area = Circle(radius=0.5, color="#FFA500", fill_opacity=0.3).move_to(phone.get_center())
        
        self.play(FadeIn(phone), FadeIn(risk_area))
        
        privacy_text = Text("Privacy Concern", color="#FF0000", font_size=20)
        self.place_at_grid(privacy_text, 'C4', scale_factor=0.7)
        self.play(Write(privacy_text))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        self.play(FadeOut(user_a), FadeOut(user_b), FadeOut(label_a), FadeOut(label_b), FadeOut(barrier), FadeOut(phone), FadeOut(risk_area), FadeOut(privacy_text))
        
        question_mark = Text("?", color="#00FFFF", font_size=72)
        self.place_at_grid(question_mark, 'C3', scale_factor=1.0)
        self.play(Write(question_mark))
        self.wait(2)
