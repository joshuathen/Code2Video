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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup title and lecture lines
        title_str = "Summary & Real-World Intuition"
        lecture_lines = [
            "Change of basis simplifies many complex mathematical problems.",
            "It's like switching languages to describe the same world.",
            "Choosing the right basis reveals hidden structures."
        ]
        self.setup_layout(title_str, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Show a 4x4 grid of gray squares (#888888) representing pixels.
        self.lecture[0].set_color(YELLOW)
        
        squares_group = VGroup(*[
            Square(side_length=0.45, fill_opacity=1, fill_color="#888888", stroke_width=1, stroke_color=WHITE)
            for _ in range(16)
        ]).arrange_in_grid(rows=4, cols=4, buff=0.1)
        
        # Applying Fix for Issue 45: Position squares_group correctly
        self.place_in_area(squares_group, 'B3', 'E6', scale_factor=0.7)
        
        self.play(FadeIn(squares_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Morph the squares into 4 different colored wave patterns (#00FFFF, #FF00FF, #FFFF00, #00FF00).
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        colors = ["#00FFFF", "#FF00FF", "#FFFF00", "#00FF00"]
        waves_group = VGroup()
        
        for i in range(4):
            # Create a wave for each color.
            wave = FunctionGraph(
                lambda x: 0.25 * np.sin(2 * PI * x + i * PI/2),
                x_range=[-1.5, 1.5],
                color=colors[i]
            )
            waves_group.add(wave)
        
        waves_group.arrange(DOWN, buff=0.4)
        
        # Applying Fix for Issue 44: Position waves_group correctly
        self.place_in_area(waves_group, 'B3', 'E6', scale_factor=0.9)
        
        self.play(ReplacementTransform(squares_group, waves_group), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Fade in text "Simpler Representation" in white (#FFFFFF).
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        simpler_text = Text("Simpler Representation", font_size=24, color="#FFFFFF")
        
        # Applying Fix for Issue 43: Position simpler_text correctly
        self.place_in_area(simpler_text, 'F3', 'F6', scale_factor=0.8)
        
        self.play(FadeIn(simpler_text))
        self.wait(2)

        # Final cleanup/end state
        self.lecture[2].set_color(WHITE)
        self.wait(1)
