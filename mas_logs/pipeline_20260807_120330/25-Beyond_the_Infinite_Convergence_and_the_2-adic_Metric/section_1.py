from manim import *
import numpy as np

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
        title = "The Paradox of the Growing Sum"
        lines = [
            "Standard sums grow without bound.",
            "Consider the series 1 plus 2 plus 4.",
            "Every step doubles the distance from zero.",
            "Digit the Rabbit jumps further and further away.",
            "Can we redefine distance to make this converge?"
        ]
        self.setup_layout(title, lines)

        # Helper for number line mapping
        # 0 -> D1 (x=0.5), 15 -> D6 (x=5.5)
        def get_nl_pos(val, row="D"):
            x = 0.5 + (val / 3.0)
            y = self.grid[f"{row}1"][1]
            return np.array([x, y, 0])

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        
        # Draw a standard number line (#FFFFFF) with labels 0, 1, 3, 7, 15.
        nl_line = Line(get_nl_pos(0), get_nl_pos(15), color=WHITE)
        ticks = VGroup()
        labels = VGroup()
        for val in [0, 1, 3, 7, 15]:
            tick = Line(UP*0.1, DOWN*0.1, color=WHITE).move_to(get_nl_pos(val))
            label = MathTex(str(val), font_size=20, color=WHITE).next_to(tick, DOWN, buff=0.1)
            ticks.add(tick)
            labels.add(label)
        
        number_line_group = VGroup(nl_line, ticks, labels)
        self.play(Create(nl_line), Create(ticks), Write(labels))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Create a yellow circle (#FFD700) representing Digit the Rabbit at position 0.
        digit = Circle(radius=0.15, color="#FFD700", fill_opacity=1.0)
        digit.move_to(get_nl_pos(0) + UP*0.15)
        self.play(FadeIn(digit))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(ORANGE)
        )
        
        # Animate the rabbit jumping sequentially to 1, 3, 7, and 15 with increasing arc heights.
        jumps = [1, 3, 7, 15]
        prev_val = 0
        for i, val in enumerate(jumps):
            # Calculate arc height based on jump distance
            dist = val - prev_val
            arc_path = ArcBetweenPoints(
                get_nl_pos(prev_val) + UP*0.15,
                get_nl_pos(val) + UP*0.15,
                radius=dist * 0.7 # Approximate arc height scaling
            )
            self.play(MoveAlongPath(digit, arc_path), run_time=0.8)
            prev_val = val
        
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(RED)
        )
        
        # Scale the rabbit and number line down as the rabbit moves off-screen to the right.
        # We'll simulate this by moving the group left and scaling it.
        scene_content = VGroup(number_line_group, digit)
        self.play(
            scene_content.animate.scale(0.5).shift(LEFT * 3),
            digit.animate.shift(RIGHT * 4), # move off-screen
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(GREEN)
        )
        
        # Display the red text 'Sum -> inf' (#FF0000) in the top right corner.
        sum_inf = MathTex(r"\text{Sum} \to \infty", color="#FF0000")
        self.place_at_grid(sum_inf, "A5", scale_factor=1.2)
        self.play(Write(sum_inf))
        
        self.wait(2)
