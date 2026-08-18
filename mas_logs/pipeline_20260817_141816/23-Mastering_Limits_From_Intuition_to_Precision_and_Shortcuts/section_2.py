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
        self.setup_layout("The Epsilon-Delta Rigor: Building the 'Safety Bubble'", [
            "We formalize limits using epsilon and delta.",
            "Epsilon defines our target error range vertically.",
            "Delta defines the required input range horizontally.",
            "A smaller epsilon forces a tighter delta.",
            "This ensures the function stays within limits."
        ])
        
        # Coordinate system
        axes = Axes(x_range=[-1, 5], y_range=[-1, 5], axis_config={"include_tip": False})
        self.place_at_grid(axes, 'D3', scale_factor=0.6)
        
        # Function curve
        func = axes.plot(lambda x: 0.2*(x-2)**3 + 2, color="#00BFFF")
        
        # L and c points
        c, L = 2, 2
        
        eps = ValueTracker(1.0)
        delta = ValueTracker(1.0)
        
        # Epsilon band
        eps_rect = always_redraw(lambda: Rectangle(
            width=6, height=2*eps.get_value(),
            color="#FF0000", fill_opacity=0.3, stroke_width=0
        ).move_to(axes.c2p(2, L)))
        
        # Delta interval
        delta_rect = always_redraw(lambda: Rectangle(
            width=2*delta.get_value(), height=6,
            color="#32CD32", fill_opacity=0.3, stroke_width=0
        ).move_to(axes.c2p(c, 2)))

        # Bubble asset
        bubble = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bubble.svg")
        self.place_at_grid(bubble, 'D3', scale_factor=0.2)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00BFFF")
        self.play(Create(axes), Create(func))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF0000")
        self.play(FadeIn(eps_rect))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#32CD32")
        self.play(FadeIn(delta_rect))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFFF00")
        self.play(eps.animate.set_value(0.5), delta.animate.set_value(0.5), run_time=2)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFA500")
        self.play(FadeIn(bubble))
        self.wait(2)
