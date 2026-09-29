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
        self.setup_layout("Conclusion and Application", [
            "PDFs indicate relative likelihood, not raw probability.",
            "We calculate probability by integrating density over intervals.",
            "PDFs are essential for quantifying real-world uncertainty."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Summarize: 'PDF = Probability Density' in #00FFFF
        summary = Text("PDF = Probability Density", font_size=32, color="#00FFFF")
        self.place_at_grid(summary, 'B3', scale_factor=0.8)
        self.play(Write(summary))
        self.play(self.lecture[0].animate.set_color("#00FFFF"))

        # === Animation for Lecture Line 2 ===
        # Connect visual shading (area) back to the mathematical result.
        # Fixed axes parameters based on B045
        axes = Axes(x_range=[-3, 3], y_range=[0, 1], x_length=4, y_length=2).scale(0.5)
        curve = axes.plot(lambda x: np.exp(-x**2 / 2) / np.sqrt(2 * np.pi), x_range=[-3, 3], color=WHITE)
        area = axes.get_area(curve, x_range=[-1, 1], color=BLUE, opacity=0.5)
        
        container = VGroup(axes, curve, area)
        self.place_in_area(container, 'C2', 'F5', scale_factor=1.0)
        
        self.play(Create(axes), Create(curve))
        self.play(FadeIn(area))
        self.play(self.lecture[1].animate.set_color("#00FFFF"))

        # === Animation for Lecture Line 3 ===
        # Display a final real-world graph with the label 'Risk Management' in #FFD700
        # Incorporate Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/dashboard.svg
        risk_label = Text("Risk Management", font_size=28, color="#FFD700")
        self.place_at_grid(risk_label, 'B5', scale_factor=0.9)
        
        dashboard_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dashboard.svg")
        dashboard_icon.next_to(risk_label, DOWN, buff=0.2)
        
        self.play(Write(risk_label), FadeIn(dashboard_icon))
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        self.wait(2)
