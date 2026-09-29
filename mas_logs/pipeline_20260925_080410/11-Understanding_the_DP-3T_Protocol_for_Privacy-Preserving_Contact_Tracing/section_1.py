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
        self.setup_layout("The Core Challenge: Privacy vs. Public Health", [
            "Contact tracing faces a privacy challenge.",
            "Public health must protect individual identities.",
            "DP-3T enables decentralized proximity tracing."
        ])
        
        # Elements
        alice = Dot(color=BLUE, radius=0.2)
        bob = Dot(color=RED, radius=0.2)
        label_alice = Text("Alice", font_size=16).next_to(alice, DOWN)
        label_bob = Text("Bob", font_size=16).next_to(bob, DOWN)
        cafe = Rectangle(color=GRAY, height=2, width=3)
        cafe_label = Text("Café", font_size=18).move_to(cafe.get_top() + DOWN*0.3)
        group = VGroup(cafe, cafe_label, alice, label_alice, bob, label_bob)

        # === Animation for Lecture Line 1 ===
        self.place_in_area(group, 'D2', 'F5', scale_factor=0.8)
        self.play(Create(cafe), Write(cafe_label))
        self.play(FadeIn(alice), Write(label_alice), FadeIn(bob), Write(label_bob))
        self.play(self.lecture[0].animate.set_color(YELLOW))

        # === Animation for Lecture Line 2 ===
        lock = VMobject().set_points_as_corners([UP, RIGHT, DOWN, LEFT, UP]).set_color(WHITE)
        self.place_at_grid(lock, 'E3', scale_factor=0.4)
        self.play(FadeIn(lock))
        self.play(self.lecture[1].animate.set_color(GREEN))

        # === Animation for Lecture Line 3 ===
        dp3t_text = Text("DP-3T", color=GOLD, font_size=32)
        self.place_at_grid(dp3t_text, 'E4', scale_factor=0.9)
        self.play(Write(dp3t_text))
        self.play(self.lecture[2].animate.set_color(BLUE))
        self.wait(2)
