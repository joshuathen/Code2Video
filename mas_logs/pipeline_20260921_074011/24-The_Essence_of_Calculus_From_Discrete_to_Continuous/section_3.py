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
        lecture_lines = ["Calculating areas of curves is inherently difficult.", "We approximate the area using tiny rectangles.", "Summing these rectangles gives the total area."]
        self.setup_layout("The Integral: Summing the Infinite", lecture_lines)
        
        # Setup Axes and Curve
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 4, 1], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: 0.5 * x**2, x_range=[0, 4], color=WHITE)
        
        # Applying layout fixes from VideoCritic
        self.place_in_area(axes, 'D1', 'F6', scale_factor=0.45)
        self.place_in_area(curve, 'D2', 'F6', scale_factor=0.45)
        
        # Load asset
        rect_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rectangle.svg")
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(axes), Create(curve))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Use rect_svg to represent the approximation
        rects = VGroup(*[rect_svg.copy().scale(0.2) for _ in range(6)])
        # Position them crudely for illustration
        for i, rect in enumerate(rects):
            rect.move_to(axes.c2p(0.5 + i * 0.5, 0.25 * (0.5+i*0.5)**2))
        
        self.play(self.lecture[1].animate.set_color("#FF00FF"))
        self.play(FadeIn(rects.set_color("#FF00FF")))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        # Increase rect density
        rects_dense = VGroup(*[rect_svg.copy().scale(0.1) for _ in range(12)])
        for i, rect in enumerate(rects_dense):
            rect.move_to(axes.c2p(0.25 + i * 0.3, 0.25 * (0.25+i*0.3)**2))
            
        self.play(ReplacementTransform(rects, rects_dense.set_color("#00FF00")))
        self.wait(2)
