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
        lecture_lines = [
            "Dirichlet's Theorem explores primes in arithmetic progressions.",
            "Primes of form a + nd exist infinitely.",
            "Numbers sorted into columns by remainder.",
            "Prime-filled columns extend infinitely.",
            "Primes never run out of destinations."
        ]
        self.setup_layout("Dirichlet's Theorem on Arithmetic Progressions", lecture_lines)
        
        # Grid setup (Numbers 0-23, arranged in 4 columns for mod 4)
        numbers = VGroup(*[Text(str(i), font_size=20) for i in range(24)])
        grid_visual = VGroup(*[numbers[i:i+6] for i in range(0, 24, 6)]).arrange(RIGHT, buff=0.5)
        # Using the specified area C2-F6 per feedback
        self.place_in_area(grid_visual, "C2", "F6", scale_factor=0.55)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF6347"))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#4682B4"))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(WHITE))
        # Highlight columns representing arithmetic progressions (e.g., 1+4n)
        progression = VGroup(grid_visual[0][1], grid_visual[1][1], grid_visual[2][1], grid_visual[3][1])
        self.play(progression.animate.set_color(YELLOW))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#32CD32"))
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFD700"))
        
        # Load asset: [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/drone.svg]
        drone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/drone.svg")
        self.place_at_grid(drone, "B2", scale_factor=0.6)
        
        self.play(FadeIn(drone))
        # Animate drone traveling along prime-marked columns
        self.play(drone.animate.move_to(self.grid["F2"]), run_time=2)
        
        # Flash intersection points
        self.play(Flash(drone.get_center(), color="#FFFFFF", line_length=0.3, num_lines=8))
