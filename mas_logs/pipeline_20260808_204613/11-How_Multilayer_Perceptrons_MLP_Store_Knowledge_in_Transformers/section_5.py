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
        self.setup_layout("Summary and Synthesis", [
            "MLPs are compressed, distributed knowledge databases.",
            "Knowledge emerges from collaborative weighted connections.",
            "It is a complex, structural collaboration."
        ])

        # Define conceptual blocks
        mapper = Circle(radius=0.5, color=BLUE, fill_opacity=0.5)
        memory = Square(side_length=0.8, color=RED, fill_opacity=0.5)
        weights = VGroup(*[Dot(color=YELLOW) for _ in range(5)]).arrange(RIGHT)
        locality = RoundedRectangle(corner_radius=0.1, height=0.6, width=1.0, color=GREEN, fill_opacity=0.5)

        # Labels
        l1 = Text("Mapper", font_size=18).next_to(mapper, DOWN)
        l2 = Text("Memory", font_size=18).next_to(memory, DOWN)
        l3 = Text("Weights", font_size=18).next_to(weights, DOWN)
        l4 = Text("Locality", font_size=18).next_to(locality, DOWN)

        group1 = VGroup(mapper, l1)
        group2 = VGroup(memory, l2)
        group3 = VGroup(weights, l3)
        group4 = VGroup(locality, l4)

        # === Animation for Lecture Line 1 ===
        self.place_at_grid(group1, 'B4', scale_factor=0.6)
        self.place_at_grid(group2, 'B6', scale_factor=0.6)
        self.play(FadeIn(group1), FadeIn(group2), self.lecture[0].animate.set_color(BLUE))

        # === Animation for Lecture Line 2 ===
        self.place_at_grid(group3, 'D4', scale_factor=0.6)
        self.place_at_grid(group4, 'D6', scale_factor=0.6)
        self.play(FadeIn(group3), FadeIn(group4), self.lecture[1].animate.set_color(YELLOW))

        # === Animation for Lecture Line 3 ===
        # Highlight collaboration
        center_conn = Line(group1.get_center(), group4.get_center(), color="#00FF00", stroke_width=4)
        self.add(center_conn)
        self.play(Create(center_conn), self.lecture[2].animate.set_color(GREEN))
        self.wait(2)
