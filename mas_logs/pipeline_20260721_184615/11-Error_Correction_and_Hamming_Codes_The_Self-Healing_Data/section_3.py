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
        title = "The Hamming Logic: Overlapping Circles"
        lines = [
            "Richard Hamming used overlapping circles for data.",
            "Data bits sit in the intersections of these circles.",
            "Each circle is managed by a specific parity bit.",
            "Flipping a data bit affects multiple parity circles.",
            "This unique intersection points directly to the error."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFF00")
        
        # Draw three overlapping circles
        # Using colors: #FF0000, #00FF00, #0000FF
        circle_red = Circle(radius=1.3, color="#FF0000", fill_opacity=0.3)
        circle_green = Circle(radius=1.3, color="#00FF00", fill_opacity=0.3)
        circle_blue = Circle(radius=1.3, color="#0000FF", fill_opacity=0.3)
        
        # Positioning for central intersection at C4
        # Red top-center, Green bottom-left, Blue bottom-right
        self.place_at_grid(circle_red, "B4")
        self.place_at_grid(circle_green, "D3")
        self.place_at_grid(circle_blue, "D5")
        
        self.play(Create(circle_red), Create(circle_green), Create(circle_blue))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFFF00")
        
        # Central data bit '1' at C4
        data_bit = Text("1", font_size=36, color="#FFFFFF")
        self.place_at_grid(data_bit, "C4", scale_factor=0.8) 
        
        self.play(Write(data_bit))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFFF00")
        
        # Parity labels P1, P2, P3
        p1 = Text("P1", font_size=28, color="#FF0000")
        p2 = Text("P2", font_size=28, color="#00FF00")
        p3 = Text("P3", font_size=28, color="#0000FF")
        
        # Fixed positions based on issues 35, 36 (Proximity Rule)
        self.place_at_grid(p1, "A4", scale_factor=0.8)
        self.place_at_grid(p2, "E3", scale_factor=0.8) 
        self.place_at_grid(p3, "E5", scale_factor=0.8) 
        
        self.play(FadeIn(p1), FadeIn(p2), FadeIn(p3))
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#FFFF00")
        
        # Central '1' flips to '0'
        new_data_bit = Text("0", font_size=36, color="#FFFFFF")
        self.place_at_grid(new_data_bit, "C4", scale_factor=0.8)
        
        # Flip and pulse circles orange (#FFA500)
        self.play(
            Transform(data_bit, new_data_bit),
            Indicate(circle_red, color="#FFA500"),
            Indicate(circle_green, color="#FFA500"),
            Indicate(circle_blue, color="#FFA500"),
            run_time=2
        )
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#FFFF00")
        
        # White arrow points to center
        # Issue 37: Move arrow to A5 for better clarity
        arrow = Arrow(start=UP+RIGHT, end=ORIGIN, color="#FFFFFF", buff=0.1)
        self.place_at_grid(arrow, "A5", scale_factor=0.8)
        
        self.play(GrowArrow(arrow))
        self.wait(2)
        
        # Final cleanup
        self.lecture[4].set_color(WHITE)
        self.wait(2)
