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
            "Track health risks without centralizing location data.",
            "The goal: Privacy-preserving exposure notification.",
            "Alice and Bob meet in a park."
        ])
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF0000"))
        
        health_text = Text("Public Health", color="#FF0000")
        privacy_text = Text("Privacy", color="#00FF00")
        self.place_at_grid(health_text, 'A2', scale_factor=0.9)
        self.place_at_grid(privacy_text, 'A5', scale_factor=0.9)
        self.play(FadeIn(health_text), FadeIn(privacy_text))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color("#00FF00"))
        
        # Using SVG asset for balance scale
        scale_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/scale.svg", color=WHITE)
        slider_group = VGroup(scale_img)
        self.place_in_area(slider_group, 'C2', 'D5', scale_factor=1.0)
        self.play(FadeIn(slider_group))
        self.play(Rotate(slider_group, angle=0.2, about_point=slider_group.get_center()), rate_func=there_and_back)
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color("#FFFF00"))
        
        alice_bob = Circle(radius=0.3, color="#FFFF00").set_fill(opacity=0.3)
        self.place_at_grid(alice_bob, 'F3')
        self.play(GrowFromCenter(alice_bob))
        self.wait(1)
