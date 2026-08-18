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
        self.setup_layout("The Translation Table: Seeing Connections", [
            "Every triplet connects power, root, and log.",
            "Power: 2 cubed is 8.",
            "Root: 8 to the 1/3 is 2.",
            "Log: log base 2 of 8 is 3."
        ])
        
        # --- Assets ---
        table_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/table.svg")
        log_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/log.svg")
        
        # --- Math Triplet ---
        power = MathTex(r"2^3 = 8", color=BLUE)
        root = MathTex(r"\sqrt[3]{8} = 2", color=GREEN)
        log = MathTex(r"\log_2(8) = 3", color=YELLOW)
        math_triplet = VGroup(power, root, log).arrange(DOWN, buff=0.5)
        
        # --- Arrows ---
        arrow_group = VGroup(
            Arrow(start=power.get_bottom(), end=root.get_top(), color=WHITE, buff=0.1),
            Arrow(start=root.get_right(), end=log.get_right(), color=WHITE, buff=0.1),
            Arrow(start=log.get_left(), end=power.get_left(), color=WHITE, buff=0.1)
        )
        
        # --- Layout Positioning ---
        self.place_in_area(math_triplet, 'A2', 'D4', scale_factor=0.8)
        self.place_at_grid(arrow_group, 'C3', scale_factor=0.9)
        self.place_in_area(table_icon, 'E3', 'F4', scale_factor=0.5)
        self.place_at_grid(log_icon, 'F6', scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(table_icon), FadeIn(math_triplet))
        self.lecture[0].set_color(ORANGE)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(BLUE)
        self.play(Indicate(power))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(GREEN)
        self.play(Indicate(root))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(YELLOW)
        self.play(Indicate(log), FadeIn(log_icon))
        
        # Final layout check
        self.play(Create(arrow_group))
        self.wait(2)
