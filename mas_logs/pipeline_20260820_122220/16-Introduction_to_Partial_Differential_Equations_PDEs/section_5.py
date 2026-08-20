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
        self.setup_layout("Summary and Real-world Impact", [
            "We combine variables, operators, and constraints.",
            "PDEs predict complex natural phenomena.",
            "They drive weather and digital effects."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Recap ODEs to PDEs transition
        self.lecture[0].set_color("#FFFFFF")
        
        v_label = Text("Variables", font_size=24).set_color(BLUE)
        o_label = Text("Operators", font_size=24).set_color(GREEN)
        c_label = Text("Constraints", font_size=24).set_color(YELLOW)
        
        self.place_at_grid(v_label, 'C2', scale_factor=0.8)
        self.place_at_grid(o_label, 'C4', scale_factor=0.8)
        self.place_at_grid(c_label, 'D3', scale_factor=0.8)
        
        self.play(FadeIn(v_label), FadeIn(o_label), FadeIn(c_label))
        
        arrows = VGroup(
            Arrow(v_label.get_right(), o_label.get_left(), buff=0.1),
            Arrow(o_label.get_bottom(), c_label.get_top(), buff=0.1)
        )
        self.play(Create(arrows))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Summarize Heat and Wave equations visual behavior
        self.lecture[1].set_color("#FF9900")
        
        thermometer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/thermometer.svg", color=RED)
        wave_line = FunctionGraph(lambda x: 0.5 * np.sin(3 * x), x_range=[-1, 1], color=BLUE)
        
        self.place_at_grid(thermometer, 'D2', scale_factor=0.7)
        self.place_at_grid(wave_line, 'D5', scale_factor=0.7)
        
        self.play(FadeIn(thermometer), FadeIn(wave_line))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Real-world examples: weather and effects
        self.lecture[2].set_color("#00FFFF")
        
        speaker_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speaker.svg", color=WHITE)
        pixel_grid = VGroup(*[Square(side_length=0.1, color=WHITE).set_fill(WHITE, opacity=0.3) for _ in range(9)])
        pixel_grid.arrange_in_grid(3, 3)
        
        self.place_at_grid(speaker_icon, 'D2', scale_factor=0.7)
        self.place_at_grid(pixel_grid, 'D5', scale_factor=0.7)
        
        self.play(FadeIn(speaker_icon), FadeIn(pixel_grid))
        self.wait(1)
