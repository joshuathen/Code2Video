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

class Section7Scene(TeachingScene):
    def construct(self):
        # Setup the layout with specific conclusion text
        self.setup_layout(
            "Conclusion: The Power of Topology", 
            [
                "Topology lets us solve 2D puzzles using higher dimensions.", 
                "Higher surfaces reveal answers hidden in simple planes.", 
                "Some geometric secrets are still waiting to be discovered."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Return of the loop and green square
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        
        # Create a stylized loop (blob)
        loop_points = [
            [1.5, 0, 0], [1.2, 1.2, 0], [0, 1.5, 0], [-1.2, 1.2, 0],
            [-1.5, 0, 0], [-1.2, -1.2, 0], [0, -1.5, 0], [1.2, -1.2, 0]
        ]
        loop = VMobject(color=WHITE).set_points_as_corners([*loop_points, loop_points[0]]).set_sheen(-0.2, direction=DR)
        loop.make_smooth()
        
        # Create the inscribed square in green
        inscribed_square = Square(side_length=1.4, color="#00FF00", stroke_width=4)
        inscribed_square.rotate(PI/8) # Stylized tilt
        
        section1_elements = VGroup(loop, inscribed_square)
        # Fix for issue 53: Position elements to avoid central overlap
        self.place_in_area(section1_elements, 'A2', 'B5', scale_factor=0.8)
        
        self.play(FadeIn(loop), Create(inscribed_square))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Möbius strip representation pulses slowly
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        
        # Create a stylized 2D Möbius strip (infinity/twisted loop look)
        mobius_strip = VGroup()
        # Top part
        strip_part1 = ParametricFunction(
            lambda t: np.array([2 * np.cos(t), 0.8 * np.sin(2 * t), 0]),
            t_range=[0, TAU],
            color="#00FFFF"
        )
        # Shift and thicken to look like a band
        strip_part2 = strip_part1.copy().shift(UP * 0.1)
        mobius_strip.add(strip_part1, strip_part2)
        
        # Fix for issue 52: Reduce scale and adjust area to avoid overlap
        self.place_in_area(mobius_strip, 'C2', 'D5', scale_factor=0.8)
        
        # Transition from 2D loop to Möbius representation
        self.play(
            FadeOut(section1_elements),
            FadeIn(mobius_strip)
        )
        
        # Pulsing effect
        self.play(
            mobius_strip.animate.scale(1.1),
            run_time=1.5,
            rate_func=there_and_back
        )
        self.play(
            mobius_strip.animate.scale(1.1),
            run_time=1.5,
            rate_func=there_and_back
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Final text fades in centrally
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        
        final_message = Text("The Power of Topology", color="#FFFFFF", font_size=36)
        # Fix for issue 54: Position in lower grid area for a balanced layout
        self.place_in_area(final_message, 'E2', 'F5', scale_factor=0.9)
        
        self.play(
            FadeOut(mobius_strip),
            FadeIn(final_message)
        )
        
        # Subtle emphasis on the final message
        self.play(final_message.animate.scale(1.05), run_time=1, rate_func=there_and_back)
        self.wait(3)
