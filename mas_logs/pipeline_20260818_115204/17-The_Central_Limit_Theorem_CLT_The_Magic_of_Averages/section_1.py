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
        lecture_lines_text = [
            "A distribution shows how data spreads.",
            "Uniform distributions have flat spreads.",
            "Skewed distributions are lopsided."
        ]
        self.setup_layout("Prerequisite: The Concept of Distributions", lecture_lines_text)
        
        # Elements for animations
        axes = Axes(x_range=[-3, 3], y_range=[0, 1], axis_config={"include_tip": False})
        normal_curve = axes.plot(lambda x: np.exp(-x**2), color="#3498DB")
        mean_label = MathTex(r"\mu", color="#E74C3C").scale(0.8)
        sigma_label = MathTex(r"\sigma", color="#E74C3C").scale(0.8)
        
        # Load asset
        data_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/data.svg", color="#F1C40F")
        data_points = VGroup(*[data_icon.copy().scale(0.1) for _ in range(20)])
        
        area = axes.get_area(normal_curve, [-1, 1], color="#2ECC71", opacity=0.3)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.place_at_grid(VGroup(axes, normal_curve), 'B3', scale_factor=0.6)
        self.play(Create(axes), Create(normal_curve))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFFFF"))
        self.place_at_grid(mean_label, 'C4', scale_factor=0.9) # Fix for issue 22
        self.place_at_grid(sigma_label, 'C5', scale_factor=0.9) # Fix for issue 23
        self.play(FadeIn(mean_label), FadeIn(sigma_label))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        self.place_at_grid(data_points, 'E3', scale_factor=1.0)
        self.play(FadeIn(data_points))
        self.place_in_area(area, 'B4', 'E6', scale_factor=0.8) # Fix for issue 21
        self.play(FadeIn(area))
        self.wait(2)
