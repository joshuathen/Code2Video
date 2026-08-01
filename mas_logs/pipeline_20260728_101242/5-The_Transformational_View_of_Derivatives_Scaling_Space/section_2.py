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
        # Data from storyboard
        title_text = "The Local Linearization Principle"
        lecture_lines = [
            "Zooming into a smooth curve reveals a line.",
            "Microscopic behavior acts like a simple multiplier.",
            "Every smooth map looks linear at high magnification."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Reference center of the right-side animation area
        area_center = (self.grid['A1'] + self.grid['F6']) / 2

        # === Animation for Lecture Line 1 ===
        # Initial state: highlight Line 1
        for line in self.lecture:
            line.set_color(GRAY)
        self.lecture[0].set_color(WHITE)
        
        # Draw curved line segment in #FFFFFF
        # Using a smooth curve with visible curvature
        curve = ParametricFunction(
            lambda t: np.array([t, 0.4 * np.sin(1.5 * t) + 0.2 * t**2, 0]),
            t_range=[-2, 2],
            color=WHITE
        ).set_stroke(width=4)
        
        # Place in right side area - Fix per Issue 25
        self.place_in_area(curve, 'B2', 'F6', scale_factor=0.8)
        
        # Ant asset [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/ant.svg]
        ant_proportion = 0.65
        ant_pos_initial = curve.point_from_proportion(ant_proportion)
        
        # Using SVGMobject for the ant as per Issue 20
        ant = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ant.svg", color=RED).scale(0.15)
        ant.move_to(ant_pos_initial)
        
        self.play(Create(curve), run_time=1.5)
        self.play(FadeIn(ant))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight Line 2
        self.play(
            self.lecture[0].animate.set_color(GRAY),
            self.lecture[1].animate.set_color(WHITE),
            run_time=0.5
        )
        
        # Zoom logic:
        # We need the stroke width to stay consistent even when zooming in.
        # However, ParametricFunction doesn't naturally keep stroke constant with .scale()
        # unless we clear the updater or use always_redraw (which we want to avoid).
        # We'll use an updater for the stroke_width only.
        curve.add_updater(lambda m: m.set_stroke(width=4))
        
        zoom_factor = 40 # High zoom to make the curve appear linear
        
        # Zoom in on the ant [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/ant.svg]
        # centering the ant in the animation area
        self.play(
            curve.animate.scale(zoom_factor, about_point=ant_pos_initial).shift(area_center - ant_pos_initial),
            ant.animate.move_to(area_center),
            run_time=4,
            rate_func=smooth
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight Line 3 with corresponding color #00FF00
        self.play(
            self.lecture[1].animate.set_color(GRAY),
            self.lecture[2].animate.set_color("#00FF00"),
            run_time=0.5
        )
        
        # Label local straight segment 'Linear Approximation' in #00FF00.
        approx_label = Text("Linear Approximation", font_size=24, color="#00FF00")
        # Place near the line - Fix per Issue 26 (E5, scale 0.6)
        self.place_at_grid(approx_label, 'E5', scale_factor=0.6)
        
        self.play(Write(approx_label))
        self.wait(2)
        
        # Cleanup updaters
        curve.clear_updaters()
