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
        lecture_lines = ["Transformers process all words at once.", 
                         "This allows for massive parallel computation.", 
                         "Much faster than word-by-word processing."]
        self.setup_layout("Conclusion: Scalability and Parallelism", lecture_lines)
        
        # Colors for lecture lines
        colors = [BLUE_B, GREEN_B, YELLOW_B]
        
        # Create visual elements
        # 3 parallel \"attention head\" processing units
        # Asset integration attempt: SVGMobject is not provided, using placeholder icon if needed or shape
        heads = VGroup(*[Circle(radius=0.5, color=BLUE_C, fill_opacity=0.3) for _ in range(3)])
        # Based on critic feedback: self.place_at_grid(heads, 'B4', scale_factor=0.7)
        self.place_at_grid(heads, 'B4', scale_factor=0.7)
        
        # Final output node
        output = Square(side_length=0.8, color=RED_C, fill_opacity=0.3)
        # Based on critic feedback: self.place_at_grid(output, 'D4', scale_factor=0.7)
        self.place_at_grid(output, 'D4', scale_factor=0.7)
        
        # Connections
        lines = VGroup(*[Line(heads[i].get_bottom(), output.get_top(), color=WHITE) for i in range(3)])

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(colors[0]), run_time=0.5)
        self.play(FadeIn(heads))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(colors[1]), run_time=0.5)
        self.play(Create(lines), FadeIn(output))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(colors[2]), run_time=0.5)
        self.play(Indicate(output))
        self.wait(2)
