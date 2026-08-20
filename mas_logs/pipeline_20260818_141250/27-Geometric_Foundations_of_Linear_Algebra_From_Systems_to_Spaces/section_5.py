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
        self.setup_layout("Synthesis and Summary", [
            "Solutions exist if vectors span the target.",
            "Non-trivial null space implies non-unique solutions.",
            "Inverse, column space, and null space are key."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Show summary of concepts using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/table.svg]
        concept_table = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/table.svg", color=WHITE)
        concept_box = RoundedRectangle(corner_radius=0.2, height=2.5, width=4.0, color=WHITE)
        concept_text = VGroup(
            Text("Target Vector", font_size=20),
            Text("∈", font_size=20),
            Text("Column Space", font_size=20, color=BLUE)
        ).arrange(DOWN)
        
        self.place_in_area(concept_box, 'A3', 'C5', scale_factor=0.6)
        self.place_in_area(concept_table, 'A3', 'C5', scale_factor=0.6)
        self.place_in_area(concept_text, 'A3', 'C5', scale_factor=0.6)
        
        self.play(Create(concept_box), FadeIn(concept_table), Write(concept_text))
        self.lecture[0].set_color(WHITE)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Connect matrix properties to geometry
        null_space_label = Text("Null Space", font_size=20, color=TEAL)
        self.place_at_grid(null_space_label, 'D4', scale_factor=0.8)
        self.play(Write(null_space_label), FadeIn(Arrow(self.grid["B4"], self.grid["D4"], color=TEAL)))
        self.lecture[1].set_color(TEAL)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Final view of full system map using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/map.svg]
        final_map = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/map.svg", color=WHITE)
        final_summary = VGroup(
            Text("Inverse", font_size=18),
            Text("Column Space", font_size=18, color=BLUE),
            Text("Null Space", font_size=18, color=TEAL)
        ).arrange(RIGHT, buff=0.5)
        
        self.place_in_area(final_summary, 'E2', 'F5', scale_factor=0.7)
        self.place_in_area(final_map, 'E2', 'F5', scale_factor=0.7)
        
        self.play(FadeIn(final_summary), FadeIn(final_map))
        self.lecture[2].set_color(WHITE)
        self.wait(2)
