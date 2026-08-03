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
        self.setup_layout(
            "Summary: The Global Knowledge Map",
            [
                "Thousands of key-value pairs exist across model layers.",
                "Early layers store patterns, while deeper layers store facts.",
                "MLPs serve as the long-term storage of the Transformer."
            ]
        )

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Create a grid of "neurons" (dots) to represent the MLP storage
        neuron_grid = VGroup()
        for r in range(8):
            for c in range(8):
                dot = Dot(radius=0.08, color="#444444")
                # Spacing dots roughly
                dot.move_to([c*0.45, r*0.45, 0])
                neuron_grid.add(dot)
        
        neuron_grid.center()
        # Resolved Issue 41: increase scale factor to 1.0 to use available space
        self.place_in_area(neuron_grid, 'A1', 'F6', scale_factor=1.0)
        
        self.play(FadeIn(neuron_grid, lag_ratio=0.05), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # Define clusters
        history_indices = [10, 11, 18, 19, 26, 27]
        science_indices = [36, 37, 44, 45, 52, 53]
        
        history_dots = VGroup(*[neuron_grid[i] for i in history_indices])
        science_dots = VGroup(*[neuron_grid[i] for i in science_indices])
        
        history_label = Text("History", font_size=16, color="#FF0000")
        science_label = Text("Science", font_size=16, color="#0000FF")
        
        # Position labels relative to the clusters
        history_label.next_to(history_dots, UP, buff=0.2)
        science_label.next_to(science_dots, DOWN, buff=0.2)

        self.play(
            history_dots.animate.set_color("#FF0000"),
            science_dots.animate.set_color("#0000FF"),
            Write(history_label),
            Write(science_label),
            run_time=1.5
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # Flash Drive representation
        drive_body = Rectangle(width=1.2, height=0.7, color="#C0C0C0", fill_opacity=1)
        drive_cap = Rectangle(width=0.4, height=0.5, color="#A0A0A0", fill_opacity=1)
        drive_cap.next_to(drive_body, RIGHT, buff=0)
        flash_drive = VGroup(drive_body, drive_cap)
        
        # Resolved Issue 42: repositioned and scaled flash_drive
        self.place_in_area(flash_drive, 'B1', 'E6', scale_factor=1.0)

        # MLP Block representation
        mlp_box = RoundedRectangle(corner_radius=0.1, width=3.0, height=1.8, color=BLUE)
        mlp_text = Text("MLP BLOCK", font_size=24, color=BLUE)
        mlp_block = VGroup(mlp_box, mlp_text)
        
        # Resolved Issue 43: repositioned and scaled mlp_block
        self.place_in_area(mlp_block, 'B1', 'E6', scale_factor=1.0)

        self.play(
            FadeOut(neuron_grid),
            FadeOut(history_label),
            FadeOut(science_label),
            FadeIn(flash_drive)
        )
        self.wait(1)
        
        self.play(
            ReplacementTransform(flash_drive, mlp_block),
            run_time=1.5
        )
        self.wait(3)
        
        # Reset color
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
