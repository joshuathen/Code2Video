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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Mapping Physics to Geometry", [
            "We transform velocities into a 2D plane.", 
            "Collisions trace a path in phase space.", 
            "Boundaries reflect the path to form circles."
        ])
        
        # Assets
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg", color=WHITE)
        mirror = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mirror.svg", color=WHITE)
        protractor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg", color=WHITE)
        
        # Create objects
        axes = Axes(x_range=[-2, 2], y_range=[-2, 2], axis_config={"include_tip": False})
        circle = Circle(radius=1.5, color=YELLOW)
        label = Text("Phase Space Path", font_size=18, color="#00FFFF")
        center_dot = Dot(color="#FF33FF")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        self.place_in_area(axes, "B3", "F6", scale_factor=0.5)
        self.place_at_grid(ruler, "A3", scale_factor=0.3)
        self.play(Create(axes), FadeIn(ruler))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        # Using a path approximation
        path = Arc(start_angle=0, angle=2*PI, radius=0.75, color=YELLOW)
        self.place_at_grid(path, "D5", scale_factor=0.7)
        self.place_at_grid(label, "E5", scale_factor=0.6)
        self.play(Create(path), FadeIn(label))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        self.place_at_grid(mirror, "C5", scale_factor=0.3)
        self.place_at_grid(circle, "D5", scale_factor=0.7)
        self.place_at_grid(center_dot, "D5", scale_factor=1.0)
        self.place_at_grid(protractor, "F5", scale_factor=0.3)
        self.play(FadeIn(mirror), ReplacementTransform(path, circle), FadeIn(center_dot), FadeIn(protractor))
