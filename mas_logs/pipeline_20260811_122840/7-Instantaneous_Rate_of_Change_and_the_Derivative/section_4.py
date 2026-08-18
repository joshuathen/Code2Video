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
        self.setup_layout("Practical Application: Motion Tracking", [
            "Cheetah position s(t) derivative is velocity s'(t).", 
            "Derivative is the rate at a specific moment.", 
            "Real-time speedometer shows this instantaneous rate."
        ])
        
        # Reveal lecture lines
        self.lecture.set_opacity(1)

        # === Animation for Lecture Line 1 ===
        # Cheetah position s(t) derivative is velocity s'(t).
        self.lecture[0].set_color("#FFA500")
        
        s_t_formula = MathTex("s(t) \\rightarrow s'(t) = v(t)", color="#FFA500")
        self.place_in_area(s_t_formula, 'B2', 'B5', scale_factor=1.0)
        self.add(s_t_formula)
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Derivative is the rate at a specific moment.
        self.lecture[1].set_color("#FF1493")
        
        cheetah = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cheetah.svg", color="#FF1493")
        self.place_at_grid(cheetah, 'C4', scale_factor=0.6)
        
        moment_label = Text("Specific Moment (t₀)", font_size=20, color="#FF1493")
        self.place_at_grid(moment_label, 'E3', scale_factor=0.8)
        
        self.add(cheetah, moment_label)
        self.play(Indicate(cheetah), Indicate(moment_label))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Real-time speedometer shows this instantaneous rate.
        self.lecture[2].set_color("#00FF00")
        
        speedometer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speedometer.svg", color="#00FF00")
        self.place_at_grid(speedometer, 'D3', scale_factor=0.8)
        
        self.add(speedometer)
        self.wait(3)
