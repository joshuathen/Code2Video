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
        self.setup_layout("Summary and Conclusion", [
            "Abstract spaces offer a universal language.", 
            "They bridge disparate data types.", 
            "Tools apply from AI to quantum."
        ])
        
        # Elements based on assets
        ai_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ai.svg")
        quantum_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/quantum.svg")
        summary_text = Text("Summary", color=WHITE)
        
        # Additional elements mentioned in storyboard/critics
        arrow = Arrow(start=LEFT, end=RIGHT, color=BLUE)
        matrix = Matrix([[1, 0], [0, 1]], left_bracket="[", right_bracket="]")
        wave = FunctionGraph(lambda t: 0.5 * np.sin(3 * t), color=YELLOW)
        v_marker = Text("V", font_size=36, color=RED)
        
        # Initial positioning (using suggested fixes from VideoCritic)
        self.place_at_grid(summary_text, 'B3', 1.0)
        self.place_at_grid(ai_icon, 'B4', 0.5)
        self.place_at_grid(arrow, 'A3', 0.8)
        self.place_in_area(matrix, 'A4', 'B6', 0.6)
        self.place_in_area(wave, 'D4', 'F6', 0.7)
        self.place_at_grid(v_marker, 'E5', 0.5)
        
        # Initially hide most, add summary/AI icon as per storyboard
        summary_group = VGroup(summary_text, ai_icon)
        self.play(FadeIn(summary_group))

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        self.play(FadeIn(arrow), FadeIn(matrix), FadeIn(wave), FadeIn(v_marker))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.play(FadeIn(quantum_icon))
        
        # Ending
        all_elements = VGroup(summary_group, arrow, matrix, wave, v_marker, quantum_icon)
        self.play(all_elements.animate.scale(0.1).set_opacity(0))
        self.wait(1)
