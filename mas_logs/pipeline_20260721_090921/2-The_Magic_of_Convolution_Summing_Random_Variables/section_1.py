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

class Section1Scene(TeachingScene):
    def construct(self):
        # Data
        title = "The Mystery of the Total Score"
        lines = [
            "Nutty and Pip both gather acorns every single day.",
            "Each squirrel's daily harvest follows its own unique distribution.",
            "How do we find the distribution of their total hoard?"
        ]
        
        self.setup_layout(title, lines)
        
        nutty_color = "#FFD700"
        pip_color = "#ADFF2F"
        total_color = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Nutty and Pip both gather acorns every single day.
        self.lecture[0].set_color(nutty_color)
        
        nutty_icon = Text("🐿️ Nutty", font_size=24, color=nutty_color)
        pip_icon = Text("🐿️ Pip", font_size=24, color=pip_color)
        
        self.place_at_grid(nutty_icon, "A2")
        self.place_at_grid(pip_icon, "A5")
        
        self.play(Write(nutty_icon), Write(pip_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Each squirrel's daily harvest follows its own unique distribution.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(pip_color)

        # Nutty's chart
        # Fixed positioning and scaling per Issue 21
        nutty_chart = BarChart(
            values=[0.2, 0.5, 0.3],
            bar_names=["1", "2", "3"],
            y_range=[0, 0.6, 0.2],
            y_length=1.5,
            x_length=2,
            x_axis_config={"font_size": 14, "label_constructor": Text},
            y_axis_config={"font_size": 14, "label_constructor": Text},
            bar_colors=[nutty_color]
        )
        self.place_in_area(nutty_chart, "B2", "C3", scale_factor=0.8)
        
        # Pip's chart
        # Fixed positioning and scaling per Issue 22
        pip_chart = BarChart(
            values=[0.4, 0.4, 0.2],
            bar_names=["1", "2", "3"],
            y_range=[0, 0.6, 0.2],
            y_length=1.5,
            x_length=2,
            x_axis_config={"font_size": 14, "label_constructor": Text},
            y_axis_config={"font_size": 14, "label_constructor": Text},
            bar_colors=[pip_color]
        )
        self.place_in_area(pip_chart, "B5", "C6", scale_factor=0.8)
        
        self.play(Create(nutty_chart), Create(pip_chart))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # How do we find the distribution of their total hoard?
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(total_color)

        # Total chart representing convolution (Sum of Nutty + Pip)
        # Probabilities calculated: 1+1=2(0.08), 3(0.28), 4(0.36), 5(0.22), 6(0.06)
        # Fixed positioning and scaling per Issue 23
        total_chart = BarChart(
            values=[0.08, 0.28, 0.36, 0.22, 0.06],
            bar_names=["2", "3", "4", "5", "6"],
            y_range=[0, 0.6, 0.2],
            y_length=1.8,
            x_length=4,
            x_axis_config={"font_size": 14, "label_constructor": Text},
            y_axis_config={"font_size": 14, "label_constructor": Text},
            bar_colors=[total_color]
        )
        self.place_in_area(total_chart, "E2", "F6", scale_factor=0.7)
        
        total_label = Text("Sum Distribution", font_size=20, color=total_color)
        # Align label manually relative to chart since place_in_area moved chart to center
        total_label.next_to(total_chart, UP, buff=0.2)

        # Add question mark flash per storyboard
        question_mark = Text("?", font_size=100, color=total_color)
        self.place_in_area(question_mark, "E2", "F6")

        self.play(Write(total_label))
        self.play(FadeIn(question_mark), run_time=0.5)
        self.play(FadeOut(question_mark), Create(total_chart), run_time=1.0)
        self.wait(2)
