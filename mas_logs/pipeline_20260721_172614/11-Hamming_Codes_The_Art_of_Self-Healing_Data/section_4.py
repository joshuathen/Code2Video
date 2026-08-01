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

class Section4Scene(TeachingScene):
    def construct(self):
        # 1. Fetch data from storyboard
        title_text = "The Hamming Logic: Overlapping Venn Diagrams"
        lecture_lines = [
            "Richard Hamming used overlapping sets for control.",
            "We place data bits within multiple circles.",
            "Each parity bit monitors a specific circle.",
            "One bit change affects multiple parity checks.",
            "This overlap creates a unique error signature."
        ]
        
        # Setup layout
        self.setup_layout(title_text, lecture_lines)

        # Colors from storyboard
        COLOR_RED = "#FF0000"
        COLOR_GREEN = "#00FF00"
        COLOR_BLUE = "#0000FF"
        HIGHLIGHT_COLOR = YELLOW

        # === Animation for Lecture Line 1 ===
        # Draw 3 overlapping circles: Red #FF0000, Green #00FF00, Blue #0000FF.
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        
        circle_r = Circle(radius=1.8, color=COLOR_RED, stroke_width=4).set_fill(COLOR_RED, opacity=0.1)
        circle_g = Circle(radius=1.8, color=COLOR_GREEN, stroke_width=4).set_fill(COLOR_GREEN, opacity=0.1)
        circle_b = Circle(radius=1.8, color=COLOR_BLUE, stroke_width=4).set_fill(COLOR_BLUE, opacity=0.1)

        # Updated positioning based on Issue 40: Shift right by 1 unit
        self.place_in_area(circle_r, 'A4', 'C5') # Top
        self.place_in_area(circle_g, 'C3', 'E4') # Bottom-Left
        self.place_in_area(circle_b, 'C5', 'E6') # Bottom-Right

        self.play(Create(circle_r), Create(circle_g), Create(circle_b), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Place data bits d1, d2, d3, d4 in the circle intersections.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_COLOR)

        # Updated positioning based on Issue 42: Adjusted data bits to center within intersections
        d4 = Text("d4", font_size=24) # Intersection of all three
        self.place_in_area(d4, 'C4', 'D5')
        
        d1 = Text("d1", font_size=24) # Red & Green
        self.place_in_area(d1, 'B3', 'C4')
        
        d2 = Text("d2", font_size=24) # Red & Blue
        self.place_in_area(d2, 'B5', 'C6')
        
        d3 = Text("d3", font_size=24) # Green & Blue
        self.place_in_area(d3, 'D4', 'E5')

        data_bits = VGroup(d1, d2, d3, d4)
        self.play(Write(data_bits))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Place parity bits p1, p2, p3 in the outer petals.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)

        # p1 moved right to match Red circle shift (A4-C5)
        p1 = Text("p1", font_size=24, color=COLOR_RED) # Red only
        self.place_in_area(p1, 'A4', 'B5')
        
        # Updated positioning based on Issue 41: Reposition parity bits within shifted circles
        p2 = Text("p2", font_size=24, color=COLOR_GREEN) # Green only
        self.place_in_area(p2, 'D3', 'E4')
        
        p3 = Text("p3", font_size=24, color=COLOR_BLUE) # Blue only
        self.place_in_area(p3, 'D5', 'E6')

        parity_bits = VGroup(p1, p2, p3)
        self.play(Write(parity_bits))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Highlight the Red circle and the 4 bits it monitors (p1, d1, d2, d4).
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(HIGHLIGHT_COLOR)

        self.play(
            circle_r.animate.set_stroke(width=10, color=HIGHLIGHT_COLOR).set_fill(opacity=0.3),
            Indicate(p1), Indicate(d1), Indicate(d2), Indicate(d4)
        )
        self.wait(2)
        self.play(
            circle_r.animate.set_stroke(width=4, color=COLOR_RED).set_fill(opacity=0.1)
        )

        # === Animation for Lecture Line 5 ===
        # Highlight the Green circle and its own set of 4 bits (p2, d1, d3, d4).
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(HIGHLIGHT_COLOR)

        self.play(
            circle_g.animate.set_stroke(width=10, color=HIGHLIGHT_COLOR).set_fill(opacity=0.3),
            Indicate(p2), Indicate(d1), Indicate(d3), Indicate(d4)
        )
        self.wait(2)
        
        # Return to base state
        self.play(
            circle_g.animate.set_stroke(width=4, color=COLOR_GREEN).set_fill(opacity=0.1),
            self.lecture[4].animate.set_color(WHITE)
        )
        self.wait(1)
