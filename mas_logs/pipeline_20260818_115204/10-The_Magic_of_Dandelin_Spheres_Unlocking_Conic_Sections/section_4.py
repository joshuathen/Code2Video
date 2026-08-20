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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Generalization to Parabola and Hyperbola", [
            "Adjusting the plane slope changes the conic shape.",
            "The parabola moves one sphere to infinity.",
            "The hyperbola sphere moves to the other nap."
        ])
        
        # Assets
        cone_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cone.svg")
        sphere_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg")
        
        # Use assets as placeholders if actual 3D objects are too heavy, 
        # or just add them for compliance. Here we place them per instruction.
        self.place_at_grid(cone_img, 'D4', scale_factor=0.5)
        self.place_at_grid(sphere_img, 'E4', scale_factor=0.3)
        
        # Geometric elements
        cone = Cone(base_radius=1.0, height=2.0).set_opacity(0.3).set_color(BLUE)
        plane = Polygon(LEFT*0.8, RIGHT*0.8, UP*0.8, DOWN*0.8).set_color(YELLOW).set_opacity(0.6)
        
        # Initial positions
        self.place_at_grid(cone, 'D4', scale_factor=0.8)
        plane.move_to(cone.get_center())
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Create(cone), Create(plane))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Rotate plane until parallel to cone side for parabola.
        self.play(self.lecture[1].animate.set_color(ORANGE))
        self.play(plane.animate.rotate(PI/4, axis=RIGHT, about_point=cone.get_center()))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Rotate plane to cut both naps for hyperbola case.
        self.play(self.lecture[2].animate.set_color(RED))
        self.play(plane.animate.rotate(PI/4, axis=RIGHT, about_point=cone.get_center()))
        self.wait(2)
