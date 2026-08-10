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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Summary and Geometric Elegance", [
            "Dandelin spheres bridge 3D space to 2D geometry.",
            "These shapes underpin our physical universe.",
            "Mathematical elegance defines orbits and satellite dishes."
        ])
        
        # Load Assets
        telescope = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/telescope.svg").set_color(WHITE)
        satellite = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/satellite.svg").set_color(WHITE)
        
        # Create shapes
        ellipse = Ellipse(width=1.5, height=0.8, color=BLUE)
        parabola = FunctionGraph(lambda x: 0.3*x**2, x_range=[-1.5, 1.5], color=GREEN)
        hyperbola = VGroup(
            FunctionGraph(lambda x: 0.5/x, x_range=[0.3, 1.5], color=RED),
            FunctionGraph(lambda x: 0.5/x, x_range=[-1.5, -0.3], color=RED)
        )
        
        conics = VGroup(ellipse, parabola, hyperbola, telescope).arrange(RIGHT, buff=0.3)
        
        # === Animation for Lecture Line 1 ===
        self.place_in_area(conics, 'A4', 'C6', scale_factor=0.4)
        self.play(FadeIn(conics), run_time=2)
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Dandelin sphere representation (using a Sphere object)
        sphere = Sphere(radius=0.3, color="#FF00FF", fill_opacity=0.5)
        self.place_at_grid(sphere, 'D4', scale_factor=0.6)
        self.play(FadeIn(sphere), run_time=1)
        self.lecture[1].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF00FF")
        # Fade out everything, show satellite
        self.play(FadeOut(conics), FadeOut(sphere), run_time=1)
        self.place_at_grid(satellite, 'C3', scale_factor=0.8)
        self.play(FadeIn(satellite), run_time=1)
        self.wait(2)
