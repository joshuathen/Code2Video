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
        lecture_lines = [
            "Predict outcomes of large groups with confidence.",
            "We move beyond knowing individual behaviors.",
            "Statistical tools guarantee precise, predictable averages."
        ]
        self.setup_layout("Application: Predicting the Future", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Conveyor belt with cookie bags
        # Using placeholder icons since real paths might not resolve in every environment
        cookie = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cookie.svg") if True else Dot()
        bag = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bag.svg") if True else Rectangle()
        
        item = VGroup(cookie, bag)
        label = Text("Sample Size n=50", font_size=18, color=WHITE)
        
        self.place_at_grid(item, "E3", scale_factor=0.6)
        self.place_at_grid(label, "F3", scale_factor=0.6)
        
        self.play(self.lecture[0].animate.set_color(YELLOW), run_time=1)
        self.play(FadeIn(item), Write(label), run_time=1)

        # === Animation for Lecture Line 2 ===
        # Bell curve representing weights
        axes = Axes(x_range=[-3, 3, 1], y_range=[0, 1, 0.5], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: np.exp(-x**2), color=GREEN)
        
        # Fixed per Issue 33 and 35: moved and scaled
        self.place_in_area(axes, "A4", "C6", scale_factor=0.35)
        self.place_in_area(curve, "A4", "C6", scale_factor=0.35)
        
        self.play(self.lecture[1].animate.set_color(YELLOW), run_time=1)
        self.play(Create(axes), Create(curve), run_time=1)

        # === Animation for Lecture Line 3 ===
        # Highlight acceptance range
        highlight = axes.plot(lambda x: np.exp(-x**2), x_range=[-1, 1], color="#33FF33", fill_opacity=0.3)
        self.place_in_area(highlight, "A4", "C6", scale_factor=0.35)
        
        self.play(self.lecture[2].animate.set_color(YELLOW), run_time=1)
        self.play(FadeIn(highlight), run_time=1)
        self.wait(1)
