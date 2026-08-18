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

class Section5Scene(ThreeDScene, TeachingScene):
    def construct(self):
        self.setup_layout("Conclusion and Application", [
            "Dandelin spheres bridge 3D and 2D.",
            "Conics are fundamental geometric properties.",
            "They appear throughout our physical universe."
        ])
        
        # Asset Loading
        planet = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/planet.svg")
        satellite = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/satellite.svg")

        # === Animation for Lecture Line 1 ===
        cone = Cone(base_radius=1.5, height=2.5, direction=DOWN).rotate(PI/4, axis=RIGHT)
        plane = NumberPlane(x_range=[-2, 2], y_range=[-2, 2]).rotate(PI/6, axis=RIGHT).shift(UP*0.5)
        ellipse = Circle(radius=0.8, color=WHITE).rotate(PI/2, axis=RIGHT).shift(UP*0.5)
        
        geometry = VGroup(cone, plane, ellipse, planet)
        self.place_in_area(geometry, 'B2', 'E5', scale_factor=0.6)
        
        grid_lines = NumberPlane(x_range=[-2, 2], y_range=[-2, 2])
        self.place_at_grid(grid_lines, 'B2', scale_factor=0.9)
        
        geometry_label = Text("Conic Section", font_size=20)
        self.place_at_grid(geometry_label, 'F2', scale_factor=0.7)

        self.play(Create(geometry), Create(grid_lines), Write(geometry_label))
        self.lecture[0].set_color(WHITE)

        # === Animation for Lecture Line 2 ===
        a_label = MathTex("a", color="#00FFFF").next_to(ellipse, RIGHT)
        b_label = MathTex("b", color="#00FFFF").next_to(ellipse, DOWN)
        c_label = MathTex("c", color="#00FFFF").next_to(ellipse, LEFT)
        measurements = VGroup(a_label, b_label, c_label)
        self.play(Write(measurements))
        self.lecture[1].set_color("#00FFFF")

        # === Animation for Lecture Line 3 ===
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES, zoom=0.8)
        self.place_at_grid(satellite, 'E6', scale_factor=0.5)
        self.play(FadeIn(satellite))
        self.lecture[2].set_color(WHITE)
        self.wait(2)
