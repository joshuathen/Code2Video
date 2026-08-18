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
        self.setup_layout("Measuring Failure: The Cost Function", [
            "Cost functions measure AI performance.",
            "High cost means the guess is wrong.",
            "Lower cost indicates closer accuracy."
        ])
        
        # Load asset
        icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        
        # Create axes for the parabola
        axes = Axes(x_range=[-2, 2, 1], y_range=[0, 4, 1], axis_config={"include_tip": False})
        parabola = axes.plot(lambda x: x**2, color="#FFFFFF")
        
        # Correct positions based on review
        self.place_in_area(axes, "B2", "F6", scale_factor=0.6)
        self.place_in_area(parabola, "B2", "F6", scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.play(Create(parabola))
        cost_label = Text("Cost Function", color="#FFFF00", font_size=20)
        self.place_at_grid(cost_label, "A5", scale_factor=0.8)
        self.play(Write(cost_label))
        # Place asset
        self.place_at_grid(icon, "A1", scale_factor=0.5)
        self.play(FadeIn(icon))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF0000")
        point = Dot(color="#FF0000", radius=0.1)
        point.move_to(axes.c2p(1.5, 2.25))
        self.add(point)
        self.play(FadeIn(point))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FF00")
        target_point = axes.c2p(0, 0)
        self.play(point.animate.move_to(target_point), run_time=2)
        
        pulsing_circle = Circle(color="#00FF00", radius=0.2).move_to(target_point)
        self.play(Create(pulsing_circle))
        self.play(Indicate(pulsing_circle, color="#00FF00"), run_time=1)
