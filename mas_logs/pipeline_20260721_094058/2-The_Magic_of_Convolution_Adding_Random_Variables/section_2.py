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

class Section2Scene(TeachingScene):
    def construct(self):
        title_str = "The Discrete Case: The 'Sum-to-K' Logic"
        lecture_lines = [
            "To find the probability that Z equals z.",
            "If X equals k, Y must be z minus k.",
            "This creates a diagonal line across our grid.",
            "Sum all probabilities of pairs along this diagonal.",
            "Moving the diagonal covers every possible sum."
        ]
        self.setup_layout(title_str, lecture_lines)
        
        # Helper to map (X, Y) to Grid Key
        # We use a 5x5 dice grid to leave room for labels
        # X: 1-5 maps to cols 2-6
        # Y: 1-5 maps to rows F-B (F is bottom, B is top)
        def dice_to_grid(x, y):
            row_map = {1: "F", 2: "E", 3: "D", 4: "C", 5: "B"}
            col_map = {1: "2", 2: "3", 3: "4", 4: "5", 5: "6"}
            return row_map[y] + col_map[x]

        # Colors for consistency
        line_colors = [YELLOW_A, GREEN_A, ORANGE, BLUE_A, PINK]

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(line_colors[0])
        
        dice_grid_bg = VGroup()
        for x in range(1, 6):
            for y in range(1, 6):
                sq = Square(side_length=0.9, stroke_width=1, color=GRAY)
                self.place_at_grid(sq, dice_to_grid(x, y))
                dice_grid_bg.add(sq)
        
        # Axis labels
        x_labels = VGroup()
        for i in range(1, 6):
            lbl = Text(str(i), font_size=18, color=WHITE)
            self.place_at_grid(lbl, "A" + str(i+1))
            x_labels.add(lbl)
            
        y_labels = VGroup()
        for i in range(1, 6):
            lbl = Text(str(i), font_size=18, color=WHITE)
            row_name = {1: "F", 2: "E", 3: "D", 4: "C", 5: "B"}[i]
            self.place_at_grid(lbl, row_name + "1")
            y_labels.add(lbl)
            
        # Titles for axes
        x_axis_label = Text("X", font_size=24, color=WHITE)
        y_axis_label = Text("Y", font_size=24, color=WHITE)
        
        # Positioning adjustment to keep within grid logic and avoid occlusion
        x_axis_label.next_to(x_labels, UP, buff=0.1)
        y_axis_label.next_to(y_labels, LEFT, buff=0.1)

        self.play(FadeIn(dice_grid_bg), Write(x_labels), Write(y_labels), Write(x_axis_label), Write(y_axis_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(line_colors[1])
        
        # Highlight a specific cell (k, z-k) in the grid with #00FF00
        k_val, y_val = 2, 2
        highlight_sq = Square(side_length=0.9, fill_opacity=0.5, fill_color="#00FF00", stroke_color="#00FF00")
        self.place_at_grid(highlight_sq, dice_to_grid(k_val, y_val))
        
        cell_tag = Text("(k, z-k)", font_size=14, color=WHITE)
        self.place_at_grid(cell_tag, dice_to_grid(k_val, y_val))
        
        self.play(FadeIn(highlight_sq), Write(cell_tag))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(line_colors[2])
        
        # Display the equation 'x + y = z' near the grid in #FFFFFF
        # Issue 20 Fix: Positioning at A2-A6 using place_in_area
        eqn = MathTex("x + y = z", color="#FFFFFF", font_size=36)
        self.place_in_area(eqn, "A2", "A6", scale_factor=0.8)

        # Draw a diagonal line across all cells where x + y = 4 (for example) in #FFA500
        # Sum = 4: (1,3), (2,2), (3,1)
        z4_cells = [(1,3), (2,2), (3,1)]
        z4_outlines = VGroup(*[
            Square(side_length=0.9, stroke_width=4, stroke_color="#FFA500").move_to(self.grid[dice_to_grid(x,y)])
            for x, y in z4_cells
        ])
        
        diag_line = Line(
            self.grid[dice_to_grid(1,3)],
            self.grid[dice_to_grid(3,1)],
            color="#FFA500", stroke_width=6
        )
        
        self.play(Write(eqn), Create(z4_outlines), Create(diag_line))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(line_colors[3])
        
        # Highlight all cells on diagonal
        z4_fills = VGroup(*[
            Square(side_length=0.9, fill_opacity=0.3, fill_color="#FFA500", stroke_width=0).move_to(self.grid[dice_to_grid(x,y)])
            for x, y in z4_cells
        ])
        
        # Update equation to show z=4
        eqn_z4 = MathTex("x + y = 4", color="#FFFFFF", font_size=36)
        self.place_in_area(eqn_z4, "A2", "A6", scale_factor=0.8)
        
        self.play(FadeIn(z4_fills), Transform(eqn, eqn_z4))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(line_colors[4])
        
        # Show moving the diagonal: Transition to z=5
        # z=5 cells: (1,4), (2,3), (3,2), (4,1)
        z5_cells = [(1,4), (2,3), (3,2), (4,1)]
        z5_outlines = VGroup(*[
            Square(side_length=0.9, stroke_width=4, stroke_color="#FFA500").move_to(self.grid[dice_to_grid(x,y)])
            for x, y in z5_cells
        ])
        z5_line = Line(
            self.grid[dice_to_grid(1,4)],
            self.grid[dice_to_grid(4,1)],
            color="#FFA500", stroke_width=6
        )
        
        # Issue 21 Fix: Positioning at A2-A6 using place_in_area
        z5_eqn = MathTex("x + y = 5", color="#FFFFFF", font_size=36)
        self.place_in_area(z5_eqn, "A2", "A6", scale_factor=0.8)

        self.play(
            Transform(z4_outlines, z5_outlines),
            Transform(diag_line, z5_line),
            Transform(eqn, z5_eqn),
            FadeOut(z4_fills),
            FadeOut(highlight_sq),
            FadeOut(cell_tag)
        )
        self.wait(2)
