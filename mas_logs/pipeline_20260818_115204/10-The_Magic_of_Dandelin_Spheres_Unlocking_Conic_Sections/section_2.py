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
        self.setup_layout("Introducing the Dandelin Spheres", [
            "Meet the magic Dandelin Spheres.", 
            "They touch the cone in a perfect circle.", 
            "They touch the plane at a single point."
        ])
        
        # Assets
        sphere_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg"
        cone_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/cone.svg"
        
        # Representations using SVGMobjects for requested assets
        cone_2d = SVGMobject(cone_path, color=BLUE)
        sphere_2d_1 = SVGMobject(sphere_path, color="#FF00FF")
        sphere_2d_2 = SVGMobject(sphere_path, color="#FF00FF")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF00FF"))
        self.place_at_grid(cone_2d, 'C4', scale_factor=1.0)
        self.place_at_grid(sphere_2d_1, 'B4', scale_factor=0.8)
        self.place_at_grid(sphere_2d_2, 'D4', scale_factor=0.8)
        self.play(Create(cone_2d), FadeIn(sphere_2d_1), FadeIn(sphere_2d_2))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFFFF"))
        circle1 = Circle(radius=0.4, color="#FFFFFF", stroke_width=2)
        circle2 = Circle(radius=0.3, color="#FFFFFF", stroke_width=2)
        self.place_at_grid(circle1, 'B4', scale_factor=0.8)
        self.place_at_grid(circle2, 'D4', scale_factor=0.8)
        self.play(Create(circle1), Create(circle2))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        point1 = Dot(color="#FFD700", radius=0.1)
        point2 = Dot(color="#FFD700", radius=0.1)
        self.place_at_grid(point1, 'A5', scale_factor=0.5)
        self.place_at_grid(point2, 'E5', scale_factor=0.5)
        self.play(FadeIn(point1), FadeIn(point2))
        
        self.wait(2)
