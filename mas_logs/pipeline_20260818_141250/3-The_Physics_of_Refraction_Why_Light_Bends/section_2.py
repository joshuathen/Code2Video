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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "The refractive index measures light speed.",
            "Light slows down in denser media.",
            "This speed change causes light to bend."
        ]
        self.setup_layout("Prerequisite: The Speed of Light in Media", lecture_lines)
        
        # Define elements (Using SVG assets as requested)
        vacuum_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/vacuum.svg", color=WHITE)
        vacuum_label = Text("Speed c", font_size=20, color=WHITE)
        water_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/water.svg", color="#00BFFF")
        water_label = Text("Speed v", font_size=20, color="#00BFFF")
        speed_comp = Text("v < c", font_size=24, color=RED)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.place_at_grid(vacuum_icon, 'B3', scale_factor=0.8)
        self.place_at_grid(vacuum_label, 'A3', scale_factor=0.7)
        self.play(FadeIn(vacuum_icon), Write(vacuum_label))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.place_at_grid(water_icon, 'E4', scale_factor=0.8)
        self.place_at_grid(water_label, 'D4', scale_factor=0.7)
        self.play(FadeIn(water_icon), Write(water_label))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        self.place_in_area(speed_comp, 'C5', 'F6', scale_factor=0.9)
        self.play(Write(speed_comp))
        self.wait(2)
