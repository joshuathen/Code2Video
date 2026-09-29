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
        self.setup_layout("Defining the Binomial Distribution", [
            "Binomial distributions count successes in n independent trials.",
            "Remember BINS: Binary, Independent, Number, Success.",
            "The number of trials n is fixed."
        ])
        
        # Assets
        n = 5
        coin_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg"
        marble_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/marble.svg"
        
        trials = VGroup(*[SVGMobject(coin_path, color=WHITE) for _ in range(n)])
        successes = VGroup(*[SVGMobject(marble_path, color=WHITE) for _ in range(n)])
        
        # Grouped together to be placed on grid
        trials_group = VGroup(*[
            VGroup(trials[i], successes[i].set_opacity(0)) 
            for i in range(n)
        ]).arrange(RIGHT, buff=0.2)
        
        # Positioning using grid
        self.place_at_grid(trials_group, "C3", scale_factor=0.6)
            
        success_indices = [1, 3] 
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(trials_group))
        self.lecture[0].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FFFF")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(ORANGE)
        
        # Highlight successes
        highlight_anims = []
        for i in success_indices:
            s_obj = trials_group[i][1]
            highlight_anims.append(s_obj.animate.set_opacity(1).set_color("#32CD32"))
            
        self.play(*highlight_anims, run_time=1.5)
        self.wait(2)
