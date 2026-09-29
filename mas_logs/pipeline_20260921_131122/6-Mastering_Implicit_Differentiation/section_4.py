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
        self.setup_layout("Application: The Robot Arm Path", [
            "Implicit differentiation finds paths for robots.",
            "Calculate slopes along curved paths.",
            "Robot direction depends on dy/dx."
        ])
        
        # Define the Folium of Descartes: x^3 + y^3 = 3xy
        def folium(t):
            return np.array([3*t/(1+t**3+0.0001), 3*t**2/(1+t**3+0.0001), 0])
        
        axes = Axes(x_range=[-2, 4, 1], y_range=[-2, 4, 1], axis_config={"include_tip": False}).scale(0.6)
        path = ParametricFunction(folium, t_range=[-10, 10], color=BLUE)
        curve_group = VGroup(axes, path)
        
        # Apply requested position
        self.place_in_area(curve_group, 'D2', 'F6', scale_factor=0.7)
        
        # Robot asset
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg").scale(0.3)
        
        # === Animation for Lecture Line 1 ===
        self.play(Create(curve_group), self.lecture[0].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Equation label
        eq = MathTex(r"x^3 + y^3 = 3xy").set_color(WHITE)
        self.place_in_area(eq, 'B2', 'C5', scale_factor=0.9)
        self.play(Write(eq), self.lecture[1].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Point
        robot.move_to(axes.c2p(1.5, 1.5))
        
        # Vector (approximate slope -1)
        vec = Vector(direction=np.array([1, -1, 0]), color=RED)
        vec.move_to(axes.c2p(1.5, 1.5), aligned_edge=LEFT)
        
        self.play(FadeIn(robot), GrowArrow(vec), self.lecture[2].animate.set_color(YELLOW))
        self.wait(2)
