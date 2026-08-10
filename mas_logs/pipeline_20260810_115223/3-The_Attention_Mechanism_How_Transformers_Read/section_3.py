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
        lecture_lines = ["Query defines what I am looking for.", "Key defines what I have to offer.", "Value is the core information retrieved."]
        self.setup_layout("The Mechanism: Queries, Keys, and Values", lecture_lines)
        
        # Load Assets
        magnifying_glass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifyingglass.svg")
        key_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/key.svg")
        treasure_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/treasure.svg")
        
        # Create Labels
        q_label = VGroup(Text("Q", color=BLUE), magnifying_glass).arrange(RIGHT).scale(0.8)
        k_label = VGroup(Text("K", color=GREEN), key_icon).arrange(RIGHT).scale(0.8)
        v_label = VGroup(Text("V", color=YELLOW), treasure_icon).arrange(RIGHT).scale(0.8)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.place_at_grid(q_label, "A3", scale_factor=0.8)
        self.play(FadeIn(q_label))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        self.place_at_grid(k_label, "B3", scale_factor=0.8)
        self.play(FadeIn(k_label))
        
        # Add connecting arrow
        arrow = Arrow(q_label.get_right(), k_label.get_left(), color=WHITE)
        self.place_in_area(arrow, "A3", "A4", scale_factor=0.8)
        self.play(Create(arrow))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.place_at_grid(v_label, "C3", scale_factor=0.8)
        self.play(FadeIn(v_label))
        
        # Grid visual objects
        grid_visuals = VGroup(q_label, k_label, v_label, arrow)
        self.place_in_area(grid_visuals, "A4", "F6", scale_factor=0.6)
        
        self.wait(2)
