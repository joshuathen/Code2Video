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
        self.setup_layout("Intuitive Hook: Sliding Windows", [
            "Convolution is a sliding window processing local data.",
            "Imagine a robot scanning a picture for edges.",
            "The window calculates a weighted sum of neighbors.",
            "This process highlights important features in images.",
            "We will explore this math step by step."
        ])
        
        # Animation Elements
        timeline = Line(start=self.grid['E4'], end=self.grid['E6'], color=WHITE)
        window = Rectangle(width=1.2, height=0.8, color="#FFCC00")
        highlight = Rectangle(width=1.2, height=0.8, color="#00FFFF", fill_opacity=0.3)
        highlight.move_to(window.get_center())
        
        # Add Asset: robot.svg
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        self.place_at_grid(robot, 'C4', scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(timeline), run_time=0.5)
        self.lecture[0].set_color("#FFCC00")

        # === Animation for Lecture Line 2 ===
        # Fix for Issue 26/41: Use place_in_area for sliding_window
        self.place_in_area(window, 'D4', 'D6', scale_factor=0.9)
        self.place_in_area(highlight, 'D4', 'D6', scale_factor=0.9)
        self.play(Create(window), FadeIn(highlight), FadeIn(robot), run_time=1)
        self.lecture[1].set_color("#FFCC00")

        # === Animation for Lecture Line 3 ===
        self.play(window.animate.move_to(self.grid['D5']), highlight.animate.move_to(self.grid['D5']), run_time=2)
        self.lecture[2].set_color("#FFCC00")

        # === Animation for Lecture Line 4 ===
        # Fade in 'Convolution is Feature Extraction'
        feature_text = Text("Convolution is Feature Extraction", font_size=20, color=WHITE)
        self.place_at_grid(feature_text, 'B4', scale_factor=0.7)
        self.play(FadeIn(feature_text), window.animate.move_to(self.grid['D6']), highlight.animate.move_to(self.grid['D6']), run_time=2)
        self.lecture[3].set_color("#FFCC00")

        # === Animation for Lecture Line 5 ===
        self.play(FadeOut(window), FadeOut(highlight), FadeOut(feature_text), FadeOut(robot), run_time=1)
        self.lecture[4].set_color("#FFCC00")
        self.wait(1)
