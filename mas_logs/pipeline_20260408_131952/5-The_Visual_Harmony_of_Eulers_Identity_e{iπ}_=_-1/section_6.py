from manim import *

# Fix: Manim's config utility can fail when folder paths contain braces like "{iπ}".
# We override the input_file to a safe string to prevent formatting errors.
config.input_file = "section_6.py"

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

class Section6Scene(TeachingScene):
    def construct(self):
        # 1. Setup Layout
        title_str = "Summary and Real-World Application"
        lecture_lines_list = [
            "Rotation and growth unify into this simple identity.",
            "This formula powers modern signals and quantum mechanics.",
            "Pure mathematical beauty meets essential real-world utility."
        ]
        self.setup_layout(title_str, lecture_lines_list)

        # Pre-create objects
        # Equation: e^{iπ} + 1 = 0
        # Fix: Replaced MathTex with Text to avoid FileNotFoundError when 'latex' is not installed in the environment.
        equation = Text("e^{iπ} + 1 = 0", font_size=72, color=WHITE)
        # Resolved Issue 48: The equation is positioned starting at Row A to use space better.
        self.place_in_area(equation, 'A2', 'C5', scale_factor=1.1)
        
        # Asset: stylized sine wave (osc.svg)
        osc_asset = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/osc.svg")
        osc_asset.set_color(BLUE_C)
        # Resolved Issue 47: Reduced scale and moved further right (Cols 3-6) to avoid obstructing lecture text.
        self.place_in_area(osc_asset, 'D3', 'F6', scale_factor=1.2)

        # === Animation for Lecture Line 1 ===
        # Highlight: Yellow
        self.play(self.lecture[0].animate.set_color(YELLOW), run_time=0.5)
        self.play(Write(equation))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight: Blue
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(BLUE_C),
            run_time=0.5
        )
        # Show stylized sine wave below the equation
        self.play(DrawBorderThenFill(osc_asset))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight: Gold
        gold_color = "#FFD700"
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(gold_color),
            run_time=0.5
        )
        
        # Pulse the entire equation in gold
        # Use there_and_back to pulse scale and color then return to original state
        self.play(
            equation.animate.set_color(gold_color).scale(1.1),
            rate_func=there_and_back,
            run_time=1.5
        )
        
        self.wait(2)
