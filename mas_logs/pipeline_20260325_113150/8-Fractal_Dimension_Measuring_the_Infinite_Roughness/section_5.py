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
        # Section Title and Lecture Lines
        title_text = "Visualizing Complexity: The Koch Snowflake"
        lecture_lines_text = [
            "The Koch Snowflake starts from a line.",
            "We replace the middle with a tent.",
            "Four new segments replace every original one.",
            "This repeating process creates an infinite perimeter.",
            "Its dimension is 1.26: rougher than a line."
        ]
        self.setup_layout(title_text, lecture_lines_text)

        # Helper to generate next stage of Koch curve points
        def get_koch_points(points):
            new_points = []
            for i in range(len(points) - 1):
                p1 = points[i]
                p2 = points[i+1]
                v = p2 - p1
                
                # Create the four segments of the Koch iteration
                q1 = p1 + v / 3
                # Rotate v/3 by 60 degrees around Z axis (equilateral triangle peak)
                cos60, sin60 = np.cos(PI/3), np.sin(PI/3)
                rot_v = np.array([
                    v[0] * cos60 - v[1] * sin60,
                    v[0] * sin60 + v[1] * cos60,
                    0
                ]) / 3
                q2 = q1 + rot_v
                q3 = p1 + 2 * v / 3
                
                new_points.extend([p1, q1, q2, q3])
            new_points.append(points[-1])
            return new_points

        # Fixed frame to ensure stability during transforms (keeps the baseline at same relative position)
        frame = Rectangle(width=4, height=2, stroke_opacity=0, stroke_width=0)

        # === Animation for Lecture Line 1 ===
        # Matching color: Green for the Koch curve mobjects
        self.lecture[0].set_color("#00FF00")
        
        # Initial segment from -1.5 to 1.5
        p0 = [np.array([-1.5, 0, 0]), np.array([1.5, 0, 0]) ]
        stage0_curve = VMobject(color="#00FF00")
        stage0_curve.set_points_as_corners(p0)
        
        # Anchor the curve using the frame for consistent grid placement
        stage0_group = VGroup(stage0_curve, frame.copy())
        self.place_in_area(stage0_group, "B2", "D5")
        
        self.play(Create(stage0_curve))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Matching color: Green
        self.lecture[1].set_color("#00FF00")
        
        # Iteration 1 (The "Tent" shape)
        p1 = get_koch_points(p0)
        stage1_curve = VMobject(color="#00FF00")
        stage1_curve.set_points_as_corners(p1)
        
        # Position using the same frame system
        stage1_group = VGroup(stage1_curve, frame.copy())
        self.place_in_area(stage1_group, "B2", "D5")
        
        self.play(ReplacementTransform(stage0_curve, stage1_curve))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Scaling parameters (Math content: White)
        self.lecture[2].set_color(WHITE)
        
        s_text = Text("S = 3", color=WHITE, font_size=32)
        n_text = Text("N = 4", color=WHITE, font_size=32)
        
        # Fixed collision at adjacent grid points A3 and A4
        self.place_in_area(s_text, 'A1', 'A3', scale_factor=0.8)
        self.place_in_area(n_text, 'A4', 'A6', scale_factor=0.8)
        
        self.play(Write(s_text), Write(n_text))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Iterative complexity (Matching color: Green)
        self.lecture[3].set_color("#00FF00")
        
        # Iteration 2
        p2 = get_koch_points(p1)
        stage2_curve = VMobject(color="#00FF00")
        stage2_curve.set_points_as_corners(p2)
        stage2_group = VGroup(stage2_curve, frame.copy())
        self.place_in_area(stage2_group, "B2", "D5")
        
        # Iteration 3
        p3 = get_koch_points(p2)
        stage3_curve = VMobject(color="#00FF00")
        stage3_curve.set_points_as_corners(p3)
        stage3_group = VGroup(stage3_curve, frame.copy())
        # Fixed Stage 3 Koch Snowflake horizontal cramping
        self.place_in_area(stage3_group, 'B1', 'D6', scale_factor=0.9)
        
        # Show growth through transformations
        self.play(ReplacementTransform(stage1_curve, stage2_curve), run_time=1.5)
        self.wait(0.5)
        self.play(ReplacementTransform(stage2_curve, stage3_curve), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Dimension calculation (Math content: White)
        self.lecture[4].set_color(WHITE)
        
        # Formula: D = log(4)/log(3) ≈ 1.26
        formula = Text("D = log(4) / log(3) ≈ 1.26", color="#FFFFFF", font_size=34)
        # Fixed dimension formula horizontal squeezing
        self.place_in_area(formula, 'E1', 'F6', scale_factor=0.8)
        
        self.play(Write(formula))
        self.wait(3)
