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

class Section5Scene(TeachingScene):
    def construct(self):
        # Setup the layout with title and lecture lines
        self.setup_layout(
            "Step-by-Step Calculation: Converting Coordinates",
            [
                "Use the inverse matrix to convert coordinates back.",
                "This tells us the vector's position in the new basis.",
                "Calculating this allows us to see through the robot's eyes."
            ]
        )

        # Colors for lecture lines and elements
        ORANGE_COLOR = "#FFA500"
        WHITE_COLOR = "#FFFFFF"
        BLUE_COLOR = "#87CEEB"
        GREEN_COLOR = "#90EE90"

        # === Animation for Lecture Line 1 ===
        # Instruction: Display the formula [v]_B = P-inverse * [v]_S in orange (#FFA500)
        self.play(self.lecture[0].animate.set_color(ORANGE_COLOR))
        
        # Using Text instead of MathTex to ensure reliability across environments
        formula = Text("[v]_B = P⁻¹ [v]_S", font_size=32, color=ORANGE_COLOR)
        # Fix Issue 35: Adjusted area for better horizontal balance
        self.place_in_area(formula, "A1", "B6", scale_factor=1.2)
        
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Instruction: Show the symbolic calculation of P-inverse and its multiplication with (2,4)
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(WHITE_COLOR)
        )

        # Matrix calculation representation
        calc = Text("P⁻¹ [2, 4]ᵀ = [3, 1]ᵀ", font_size=28, color=WHITE_COLOR)
        # Fix Issue 36: Expanded area to avoid cramped appearance
        self.place_in_area(calc, "C1", "D6", scale_factor=1.1)

        self.play(Write(calc))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Instruction: Update visual to show vector (2,4) at (3,1) in tilted grid
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(WHITE_COLOR)
        )

        # Create a tilted grid to represent the Robot's basis B
        # Basis vectors for the robot (example values)
        b1_vec = np.array([0.6, 0.2, 0])
        b2_vec = np.array([-0.2, 0.5, 0])
        
        robot_grid = VGroup()
        for i in range(-4, 5):
            # Lines parallel to b2
            line1 = Line(i*b1_vec - 3*b2_vec, i*b1_vec + 3*b2_vec, stroke_width=1, color=GRAY, stroke_opacity=0.3)
            # Lines parallel to b1
            line2 = Line(i*b2_vec - 3*b1_vec, i*b2_vec + 3*b1_vec, stroke_width=1, color=GRAY, stroke_opacity=0.3)
            robot_grid.add(line1, line2)
            
        robot_basis_v1 = Arrow(ORIGIN, b1_vec, buff=0, color=BLUE_COLOR, stroke_width=3)
        robot_basis_v2 = Arrow(ORIGIN, b2_vec, buff=0, color=GREEN_COLOR, stroke_width=3)
        
        robot_labels = VGroup(
            Text("b₁", font_size=18, color=BLUE_COLOR).next_to(b1_vec, RIGHT, buff=0.1),
            Text("b₂", font_size=18, color=GREEN_COLOR).next_to(b2_vec, LEFT, buff=0.1)
        )

        # The vector (2,4) in standard basis is (3,1) in this robot basis
        # v = 3*b1 + 1*b2
        v_target_coords = (3, 1)
        v_target_pos = v_target_coords[0] * b1_vec + v_target_coords[1] * b2_vec
        v_arrow = Arrow(ORIGIN, v_target_pos, buff=0, color=ORANGE_COLOR, stroke_width=5)
        v_coords_label = Text(f"({v_target_coords[0]}, {v_target_coords[1]})_B", font_size=20, color=ORANGE_COLOR).next_to(v_target_pos, UR, buff=0.1)

        robot_basis_group = VGroup(robot_grid, robot_basis_v1, robot_basis_v2, robot_labels, v_arrow, v_coords_label)
        
        # Fix Issue 37: Scale factor 0.8 and area E1-F6
        self.place_in_area(robot_basis_group, "E1", "F6", scale_factor=0.8)
        
        self.play(Create(robot_grid), run_time=1.5)
        self.play(
            Create(robot_basis_v1), 
            Create(robot_basis_v2), 
            Write(robot_labels)
        )
        self.play(
            Create(v_arrow),
            Write(v_coords_label)
        )
        
        self.wait(3)
