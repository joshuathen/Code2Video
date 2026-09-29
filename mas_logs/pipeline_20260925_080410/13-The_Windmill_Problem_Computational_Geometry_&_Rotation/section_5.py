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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Summary and Application", ["The Windmill strategy is efficient to compute.", "It helps robots navigate complex environments.", "Sensors use this to cover every area."])
        
        # === Animation for Lecture Line 1 ===
        # Summarize key points: Sweep, Pivot, Invariant.
        pts = VGroup(
            Text("Sweep", color=BLUE),
            Text("Pivot", color=GREEN),
            Text("Invariant", color=YELLOW)
        ).arrange(DOWN, aligned_edge=LEFT)
        self.place_in_area(pts, 'A4', 'C6', scale_factor=0.6)
        self.play(Write(pts))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        # Display a real-world windmill icon labeled 'W' in #FFA500.
        windmill_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/windmill.svg")
        windmill_svg.set_color("#FFA500")
        label = Text("W", color="#FFA500").next_to(windmill_svg, UP)
        icon_group = VGroup(windmill_svg, label)
        self.place_at_grid(icon_group, 'D2', scale_factor=0.8)
        self.play(FadeIn(icon_group))
        self.lecture[1].set_color(GREEN)

        # === Animation for Lecture Line 3 ===
        # End with the final state labeled 'Invariant' in #FFFFFF.
        final_text = Text("Invariant", color=WHITE)
        self.place_at_grid(final_text, 'D4', scale_factor=0.9)
        self.play(Write(final_text))
        self.lecture[2].set_color(YELLOW)
        
        self.wait(2)
