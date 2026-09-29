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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Hook & The Prerequisite: Light's Speed in Vacuum", 
                          ["Why does a straw look broken in water?", 
                           "Light travels at constant speed in a vacuum.", 
                           "This speed is our baseline, 'c'."])
        
        # --- Pre-calculate elements ---
        water_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/water.svg")
        glass_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/glass.svg")
        
        # 1. Straw in water hook
        water_surface = Rectangle(width=4, height=2, color=BLUE_D, fill_opacity=0.3)
        straw = Line(start=UP*1.5, end=DOWN*1.5, color=ORANGE, stroke_width=6)
        straw_in_water = VGroup(water_surface, straw, water_icon)
        self.place_in_area(straw_in_water, 'B4', 'E5', scale_factor=0.6)

        # 2. Light in vacuum
        light_beam = Line(start=LEFT*2, end=RIGHT*2, color=WHITE, stroke_width=4)
        self.place_at_grid(light_beam, 'D3', scale_factor=0.7)

        # 3. Speed of light
        speed_text = MathTex(r"c = 3 \times 10^8 \, m/s", color=YELLOW)
        self.place_at_grid(speed_text, 'E4', scale_factor=0.75)
        
        # Glass boundary
        boundary = VGroup(glass_icon)

        # --- Animations ---
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.play(Create(straw_in_water), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.play(FadeOut(straw_in_water), Create(light_beam), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.play(Write(speed_text), Create(boundary), run_time=2)
        self.wait(1)
