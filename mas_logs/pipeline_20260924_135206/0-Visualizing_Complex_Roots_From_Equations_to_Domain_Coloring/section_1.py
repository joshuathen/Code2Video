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
        lecture_lines = [
            "Complex numbers are points in a 2D plane.",
            "Functions map inputs to new output points.",
            "Observe the rubber sheet transformation."
        ]
        self.setup_layout("Prerequisites: The Complex Plane", lecture_lines)
        
        # Animations
        # Use SVG for background
        plane_bg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sheet.svg")
        plane = ComplexPlane(x_range=[-3, 3], y_range=[-3, 3])
        # Group them if needed, but for now apply grid to plane
        self.place_at_grid(plane, 'E4', scale_factor=0.6)
        
        unit_circle = Circle(radius=0.5, color="#FF00FF")
        unit_circle.move_to(plane.get_center())
        
        origin_label = Text("0", color="#FFFF00", font_size=20)
        self.place_at_grid(origin_label, 'D5', scale_factor=0.3)
        
        real_axis = Text("Re", color="#00FFFF", font_size=20)
        self.place_at_grid(real_axis, 'E6', scale_factor=0.4)
        
        imag_axis = Text("Im", color="#B6FF00", font_size=20)
        self.place_at_grid(imag_axis, 'C4', scale_factor=0.4)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF0000"))
        self.play(Create(plane), Create(unit_circle))
        self.play(Write(origin_label), Write(real_axis), Write(imag_axis))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#0000FF"))
        self.play(plane.animate.apply_complex_function(lambda z: z**2))
