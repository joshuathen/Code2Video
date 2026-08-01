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
        # Setup the layout with title and lecture lines
        self.setup_layout("The Big Picture: Static vs. Dynamic", [
            "Algebra handles static shapes and constant speeds perfectly.",
            "But the real world is curvy and constantly changing.",
            "Calculus measures change at a specific, exact moment."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Show a square (#00FF00) and a car icon [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/car.png] (#FFFF00) moving at constant speed. 
        # Label: 'Algebra: Constant'. Change lecture line 1 color to #FFFF00.
        
        line1_color = "#FFFF00"
        square = Square(side_length=1.5, color="#00FF00", fill_opacity=0.3)
        self.place_at_grid(square, "B3")
        
        # Use ImageMobject for car icon as per Issue 22
        car1 = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/car.png")
        self.place_at_grid(car1, "C2", scale_factor=0.3)
        
        label1 = Text("Algebra: Constant", font_size=20, color=WHITE)
        # Apply Issue 26: Positioning fix for label1
        self.place_in_area(label1, "A2", "A4", scale_factor=0.8)
        
        self.play(
            self.lecture[0].animate.set_color(line1_color),
            Create(square),
            FadeIn(car1),
            Write(label1)
        )
        self.wait(0.5)
        
        # Constant speed movement to grid C5
        self.play(car1.animate.move_to(self.grid["C5"]), run_time=2, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show a curve (#00FFFF) with a car icon [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/car.png] (#FFFF00) accelerating. 
        # Label: 'Calculus: Changing'. Change lecture line 2 color to #FFFF00.
        
        # Create a quadratic curve for accelerating appearance
        curve = ParametricFunction(
            lambda t: np.array([t, 0.2 * t**2, 0]),
            t_range=[0, 3],
            color="#00FFFF"
        )
        # Place the curve in the designated area
        self.place_in_area(curve, "D2", "F5")
        
        label2 = Text("Calculus: Changing", font_size=20, color=WHITE)
        # Apply Issue 27: Positioning fix for label2 to avoid overlap
        self.place_in_area(label2, "C2", "C4", scale_factor=0.8)
        
        car2 = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/car.png")
        car2.scale(0.3)
        # Align car2 with start of curve
        car2.move_to(curve.get_start())
        
        self.play(
            self.lecture[1].animate.set_color("#FFFF00"),
            Create(curve),
            FadeIn(car2),
            Write(label2)
        )
        
        # Acceleration movement along the curve
        self.play(
            MoveAlongPath(car2, curve),
            run_time=2,
            rate_func=rate_functions.ease_in_quad
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Flash a single point on the curve and draw a tangent line (#FF8C00). 
        # Label: 'Instantaneous Speed'. Change lecture line 3 color to #FFFF00.
        
        # Select a point on the curve (alpha is proportion 0.0 to 1.0)
        alpha_point = 0.7
        point_on_curve = curve.point_from_proportion(alpha_point)
        dot = Dot(point_on_curve, color="#FF8C00")
        
        # Tangent line at that point
        tangent = TangentLine(curve, alpha=alpha_point, length=2.5, color="#FF8C00")
        
        label3 = Text("Instantaneous Speed", font_size=20, color=WHITE)
        # Apply Issue 28: Positioning fix for label3
        self.place_in_area(label3, "F2", "F4", scale_factor=0.7)
        
        self.play(
            self.lecture[2].animate.set_color("#FFFF00"),
            Flash(dot, color="#FF8C00"),
            Create(dot),
            Create(tangent),
            Write(label3)
        )
        self.wait(2)
