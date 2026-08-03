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

class Section2Scene(TeachingScene):
    def construct(self):
        # Title and Lecture Lines
        title = "Prerequisite: The Ice Cream Cone Theorem"
        lines = [
            "Imagine two lines tangent to a sphere from one point.",
            "These two tangent segments are always equal in length.",
            "This simple geometric fact is our most powerful tool."
        ]
        self.setup_layout(title, lines)
        
        # === Animation for Lecture Line 1 ===
        # Display a sphere [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg] (#FFFFFF) and a point P (#FF4500) outside of it.
        sphere = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg").set_color(WHITE)
        self.place_in_area(sphere, 'B2', 'E5')
        
        # [Issue 31] Point P and its label are too close to the lecture notes.
        # Fix: Move to C2 and scale.
        point_p = Dot(color="#FF4500")
        self.place_at_grid(point_p, 'C2', scale_factor=0.8)
        label_p = MathTex("P", color="#FF4500", font_size=24).next_to(point_p, LEFT, buff=0.1)
        
        self.lecture[0].set_color(YELLOW)
        self.play(DrawBorderThenFill(sphere), FadeIn(point_p), Write(label_p))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Draw two tangent segments from P to points A and B (#00FF00) on the sphere's surface.
        sphere_center = sphere.get_center()
        # Area B2 to E5 is 3 units wide/high, so approximate radius is 1.5.
        radius = 1.5
        p_pos = point_p.get_center()
        
        # Geometry for tangent points from P to Circle
        dist = np.linalg.norm(p_pos - sphere_center)
        angle_p = np.arctan2(p_pos[1] - sphere_center[1], p_pos[0] - sphere_center[0])
        
        # Safety check for arccos
        if radius >= dist:
            radius = dist * 0.95
            
        alpha = np.arccos(radius / dist)
        
        pos_a = sphere_center + np.array([
            radius * np.cos(angle_p + alpha),
            radius * np.sin(angle_p + alpha),
            0
        ])
        pos_b = sphere_center + np.array([
            radius * np.cos(angle_p - alpha),
            radius * np.sin(angle_p - alpha),
            0
        ])
        
        dot_a = Dot(pos_a, color="#00FF00")
        dot_b = Dot(pos_b, color="#00FF00")
        label_a = MathTex("A", color="#00FF00", font_size=24).next_to(dot_a, UP + RIGHT, buff=0.1)
        label_b = MathTex("B", color="#00FF00", font_size=24).next_to(dot_b, DOWN + RIGHT, buff=0.1)
        
        line_pa = Line(p_pos, pos_a, color="#00FF00")
        line_pb = Line(p_pos, pos_b, color="#00FF00")
        
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.play(
            Create(line_pa), 
            Create(line_pb), 
            FadeIn(dot_a), 
            FadeIn(dot_b),
            Write(label_a),
            Write(label_b)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight segments PA and PB, and display the equation 'PA = PB' to show equality.
        equation = MathTex("PA = PB", color="#00FF00", font_size=32)
        # [Issue 32] Equation 'PA = PB' is placed at A4, making it appear small and disconnected.
        # Fix: place in area 'A3' to 'A5'.
        self.place_in_area(equation, 'A3', 'A5', scale_factor=1.0)
        
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        self.play(
            line_pa.animate.set_stroke(width=6),
            line_pb.animate.set_stroke(width=6),
            Write(equation)
        )
        self.play(
            line_pa.animate.set_stroke(width=2),
            line_pb.animate.set_stroke(width=2)
        )
        self.wait(2)
