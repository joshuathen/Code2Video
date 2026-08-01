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
        # Setup layout with title and lecture lines from storyboard
        self.setup_layout("Summary and Application", [
            "We moved from histograms to smooth PDF curves.",
            "PDFs use area to find probability in intervals.",
            "These curves model real-world data like failure times."
        ])

        # Color definitions
        COLOR_NORMAL = BLUE
        COLOR_EXP = GREEN
        COLOR_DANGER = "#FF0000" # Red
        COLOR_GOLD = "#FFD700"   # Gold

        # === Animation for Lecture Line 1 ===
        # Script: "We moved from histograms to smooth PDF curves."
        self.lecture[0].set_color(COLOR_NORMAL)
        
        axes = Axes(
            x_range=[-3, 3, 1], 
            y_range=[0, 0.5, 0.1], 
            x_length=5, 
            y_length=4, 
            axis_config={"include_tip": False}
        )
        # Fix Issue 34: Scaled axes to 0.7 to avoid crowding bottom text.
        self.place_in_area(axes, 'A1', 'E6', scale_factor=0.7)
        
        # Normal distribution PDF: mu=0, sigma=1
        normal_curve = axes.plot(
            lambda x: np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi), 
            color=COLOR_NORMAL
        )
        
        self.play(Create(axes), Create(normal_curve), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Script: "PDFs use area to find probability in intervals."
        self.lecture[1].set_color(COLOR_DANGER)
        
        # Exponential distribution PDF: lambda=0.4 (to match scale and peak roughly)
        exp_curve = axes.plot(
            lambda x: 0.4 * np.exp(-0.4 * x), 
            x_range=[0, 3], 
            color=COLOR_EXP
        )
        
        # Morph the curve between Normal and Exponential shapes
        self.play(ReplacementTransform(normal_curve, exp_curve), run_time=1.5)
        self.wait(0.5)
        
        # Shade a small 'danger zone' tail in red (#FF0000) at the end of the curve.
        # Tail interval: x from 2 to 3
        danger_area = axes.get_area(exp_curve, x_range=[2, 3], color=COLOR_DANGER, opacity=0.5)
        danger_label = Text("Danger Zone", font_size=20, color=COLOR_DANGER)
        # Fix Issue 33: Scaled danger_label to 0.6 to reduce visual clutter near axis.
        self.place_at_grid(danger_label, 'D5', scale_factor=0.6)
        
        self.play(FadeIn(danger_area), Write(danger_label))
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # Script: "These curves model real-world data like failure times."
        self.lecture[2].set_color(COLOR_GOLD)
        
        # Display the final summary text in gold
        final_summary = Text("PDF: The Logic of Continuous Data", font_size=24, color=COLOR_GOLD)
        # Fix Issue 32: Scaled final_summary to 0.8 to prevent cramping against screen edge.
        self.place_in_area(final_summary, 'F1', 'F6', scale_factor=0.8)
        
        self.play(Write(final_summary))
        self.wait(3)
