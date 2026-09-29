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
        lecture_lines_text = ["The sum diverges when s is 1 or less.", "Analytic continuation extends the function's reach.", "Think of it as mapping unknown territory."]
        self.setup_layout("Analytic Continuation: Looking Beyond the Horizon", lecture_lines_text)
        
        # Create visual elements
        axes = Axes(x_length=4, y_length=3, x_range=[-2, 2], y_range=[-1, 3], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: 1/x if x > 0 else 0, x_range=[0.3, 2], color=WHITE)
        domain_label = Text("Original Domain", font_size=18, color=WHITE)
        
        # Load SVG Assets
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        globe = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/globe.svg")
        
        # Initial positions
        self.place_in_area(axes, 'B3', 'E6', scale_factor=0.6)
        self.place_at_grid(curve, 'D4', scale_factor=0.7)
        self.place_at_grid(domain_label, 'F2')
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(axes), Create(curve), FadeIn(domain_label))
        self.play(self.lecture[0].animate.set_color("#FF6347")) # Tomato
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        extended_curve = axes.plot(lambda x: np.sin(x*3)+1.5, x_range=[-2, 2], color="#00BFFF")
        ext_label = Text("Analytic Continuation", font_size=18, color="#00BFFF")
        
        self.play(self.lecture[1].animate.set_color("#00BFFF"))
        self.play(Create(extended_curve))
        self.place_at_grid(ext_label, 'B5', scale_factor=0.7)
        self.place_at_grid(compass, 'A1', scale_factor=0.5)
        self.play(FadeIn(ext_label), FadeIn(compass))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        highlight_region = Rectangle(width=1.6, height=1.6, color="#90EE90", fill_opacity=0.3)
        self.place_at_grid(highlight_region, 'D3', scale_factor=0.8)
        self.place_at_grid(globe, 'D3', scale_factor=0.3)
        
        self.play(self.lecture[2].animate.set_color("#90EE90"))
        self.play(FadeIn(highlight_region), FadeIn(globe))
        self.wait(2)
