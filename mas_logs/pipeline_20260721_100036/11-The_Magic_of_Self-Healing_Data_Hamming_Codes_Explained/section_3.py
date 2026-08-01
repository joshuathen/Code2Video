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
        # Data from storyboard
        lecture_lines = [
            "Richard Hamming used multiple overlapping parity bits.",
            "Each parity bit covers a specific subset of data.",
            "We place bits in positions that are powers of two.",
            "Overlapping circles help us visualize these data groups.",
            "Every data bit is now watched by multiple circles."
        ]
        self.setup_layout("The Hamming Strategy: Overlapping Circles", lecture_lines)

        # Colors from instructions
        RED_COLOR = "#FF0000"
        GREEN_COLOR = "#00FF00"
        BLUE_COLOR = "#0000FF"
        DATA_COLOR = "#FFFF00"
        PARITY_COLOR = "#FFFFFF"
        PARITY_LABEL_COLOR = "#ADD8E6"
        DATA_LABEL_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Richard Hamming used multiple overlapping parity bits.
        self.lecture[0].set_color(WHITE)
        
        # Define the three overlapping circles
        circle_red = Circle(radius=1.5, color=RED_COLOR, stroke_width=4, fill_opacity=0.1)
        circle_green = Circle(radius=1.5, color=GREEN_COLOR, stroke_width=4, fill_opacity=0.1)
        circle_blue = Circle(radius=1.5, color=BLUE_COLOR, stroke_width=4, fill_opacity=0.1)

        # Positions calculated to overlap mathematically on the 6x6 grid
        self.place_at_grid(circle_red, 'C3')
        self.place_at_grid(circle_green, 'C5')
        self.place_at_grid(circle_blue, 'E4')

        self.play(Create(circle_red), Create(circle_green), Create(circle_blue), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Each parity bit covers a specific subset of data.
        self.lecture[1].set_color(WHITE)
        # Sequential highlighting of the groups
        self.play(circle_red.animate.set_fill(opacity=0.3), run_time=0.6)
        self.play(circle_red.animate.set_fill(opacity=0.1), circle_green.animate.set_fill(opacity=0.3), run_time=0.6)
        self.play(circle_green.animate.set_fill(opacity=0.1), circle_blue.animate.set_fill(opacity=0.3), run_time=0.6)
        self.play(circle_blue.animate.set_fill(opacity=0.1), run_time=0.6)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # We place bits in positions that are powers of two.
        self.lecture[2].set_color(PARITY_LABEL_COLOR)
        
        # Parity bits (1, 2, 4)
        # Each bit has a dot and a label
        def create_bit_unit(num_str, bit_color, label_color):
            bit = Dot(radius=0.08, color=bit_color)
            label = Text(num_str, font_size=18, color=label_color)
            return VGroup(bit, label).arrange(DOWN, buff=0.1)

        v1 = create_bit_unit("1", PARITY_COLOR, PARITY_LABEL_COLOR)
        v2 = create_bit_unit("2", PARITY_COLOR, PARITY_LABEL_COLOR)
        v4 = create_bit_unit("4", PARITY_COLOR, PARITY_LABEL_COLOR)

        self.place_at_grid(v1, 'B2') # Red only
        self.place_at_grid(v2, 'B6') # Green only
        # Issue 30 Fix: Change v4 to 'E4' and scale 0.8
        self.place_at_grid(v4, 'E4', scale_factor=0.8) # Blue only (lowered to balance)

        self.play(FadeIn(v1), FadeIn(v2), FadeIn(v4))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Overlapping circles help us visualize these data groups.
        self.lecture[3].set_color(WHITE)
        
        # Data bits at intersections: 3, 5, 6, 7
        v3 = create_bit_unit("3", DATA_COLOR, DATA_LABEL_COLOR)
        v5 = create_bit_unit("5", DATA_COLOR, DATA_LABEL_COLOR)
        v6 = create_bit_unit("6", DATA_COLOR, DATA_LABEL_COLOR)
        v7 = create_bit_unit("7", DATA_COLOR, DATA_LABEL_COLOR)

        # Issue 28 Fix: Change v3 to 'A4' and scale 0.8
        self.place_at_grid(v3, 'A4', scale_factor=0.8) # Red + Green (Peak)
        self.place_at_grid(v5, 'D3') # Red + Blue
        self.place_at_grid(v6, 'D5') # Green + Blue
        # Issue 29 Fix: Change v7 scale to 0.9
        self.place_at_grid(v7, 'D4', scale_factor=0.9) # Red + Green + Blue

        self.play(FadeIn(v3), FadeIn(v5), FadeIn(v6), FadeIn(v7))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Every data bit is now watched by multiple circles.
        self.lecture[4].set_color(DATA_COLOR)
        
        # Pulse all data bits to emphasize they are 'watched'
        self.play(
            v3.animate.scale(1.3),
            v5.animate.scale(1.3),
            v6.animate.scale(1.3),
            v7.animate.scale(1.3),
            run_time=0.8,
            rate_func=there_and_back
        )
        self.wait(2)
