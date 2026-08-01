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
        # Fetching data from storyboard
        title_text = "The 'Power' in Action"
        lecture_lines = [
            "- Derivative rules let us find speed at any time.",
            "- For distance t squared, the speed is 2t.",
            "- Calculus turns static graphs into dynamic measurements."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Display the function f(t) = t^2 in #3498DB at the top of the screen.
        self.play(self.lecture[0].animate.set_color("#3498DB"))
        f_t = MathTex("f(t) = t^2", color="#3498DB")
        # Fix Issue 32: scale_factor from 1.5 to 1.2
        self.place_in_area(f_t, 'B2', 'B5', scale_factor=1.2)
        self.play(Write(f_t))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Animate a transition where f(t) = t^2 becomes f'(t) = 2t in #2ECC71.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#2ECC71")
        )
        df_t = MathTex("f'(t) = 2t", color="#2ECC71")
        # Fix Issue 31: Move from D2-D5 to C2-C5 and scale to 1.2
        # Fix Issue 32 (implied reduction for df_t as well): scale_factor to 1.2
        self.place_in_area(df_t, 'C2', 'C5', scale_factor=1.2)
        # Using a copy to transform f_t into df_t while keeping the visual logic
        self.play(TransformFromCopy(f_t, df_t))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show the calculation 'At t = 3, Speed = 2(3) = 6 m/s' appearing in #FFFFFF.
        # accompanied by the speed icon [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/speed.svg]
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FFFFFF")
        )
        calc_text = MathTex(r"\text{At } t = 3, \text{ Speed} = 2(3) = 6 \text{ m/s}", color="#FFFFFF")
        # Load asset (Issue 19)
        speed_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speed.svg", height=0.6, color=WHITE)
        
        calc_group = VGroup(calc_text, speed_icon).arrange(RIGHT, buff=0.3)
        # Fix Issue 33: Move from F1-F6 to E1-E6
        self.place_in_area(calc_group, 'E1', 'E6', scale_factor=0.8)
        
        self.play(FadeIn(calc_group, shift=UP))
        self.wait(2)
        
        # Final color reset
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
