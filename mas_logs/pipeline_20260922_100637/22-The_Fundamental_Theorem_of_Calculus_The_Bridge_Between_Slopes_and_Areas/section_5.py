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
        self.setup_layout("Summary & Synthesis", ["Two problems are now unified.", "Summing slivers equals finding antiderivatives.", "Calculus connects slopes and areas."])
        
        # Elements
        concept_box = RoundedRectangle(corner_radius=0.2, color="#FF5733", height=1.5, width=4)
        concept_text = Text("FTC Relationship", font_size=24, color="#FF5733")
        
        accumulation_visual = VGroup(
            Square(color="#33FF57", fill_opacity=0.3),
            Circle(color="#33FF57", radius=0.3),
            Line(LEFT*0.5, RIGHT*0.5, color="#33FF57")
        ).arrange(DOWN)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF5733")
        self.place_at_grid(concept_box, "B4", scale_factor=0.8)
        self.place_at_grid(concept_text, "B4", scale_factor=0.7)
        self.play(FadeIn(concept_box), Write(concept_text))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF5733")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#33FF57")
        self.place_in_area(accumulation_visual, "D4", "E5", scale_factor=0.7)
        self.play(FadeIn(accumulation_visual))
        self.wait(2)
