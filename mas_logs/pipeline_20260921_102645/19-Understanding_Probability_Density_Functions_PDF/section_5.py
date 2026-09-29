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
            "PDFs describe probability density, not probability.",
            "Clustered outcomes create tall peaks.",
            "Spread outcomes result in flat curves."
        ]
        self.setup_layout("Conclusion & Key Takeaways", lecture_lines)
        
        # Animations
        # Create persistent axes and curve
        axes = Axes(x_range=[-2, 2, 1], y_range=[0, 1.5, 0.5], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: np.exp(-x**2), color=BLUE)
        area = axes.get_area(curve, x_range=[-1.5, 1.5], color=BLUE, opacity=0.3)
        pdf_group = VGroup(axes, curve, area)
        self.place_in_area(pdf_group, 'C1', 'F6', scale_factor=0.65)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        self.play(Create(axes), Create(curve), FadeIn(area))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF6347"))
        tall_curve = axes.plot(lambda x: 1.2 * np.exp(-4 * x**2), color=RED)
        self.play(Transform(curve, tall_curve))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#32CD32"))
        flat_curve = axes.plot(lambda x: 0.6 * np.exp(-0.5 * x**2), color=GREEN)
        self.play(Transform(curve, flat_curve))
        
        self.wait(2)
